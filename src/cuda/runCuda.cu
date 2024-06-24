#include "runCuda.cuh"
#include "cudaInitData.cuh"
#include "lbmKernel2D.cuh"
#include "cudaIsosurface.cuh"
#include "cudaUtils.cuh"
#include "ui/app.h"
#include "io/writer.h"
#include "initData.h"
#include "graphics/OpenGL/OpenGLWindow.h"
#include "graphics/renderer.h"
#include "graphics/texture.h"
#include "ui/renderLayer.h"
#include "utils.h"


#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
//This need to be after glad and glfw
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <vector_functions.h>
#include <iostream>
#include <memory>
#include <unistd.h>
#include <vector>
#include <type_traits>
#include <chrono>

template<typename PRECISION>
void FLB::h_launchCudaCalculations2D(FLB::OptionsCalculation& optionsCalc, std::vector<PRECISION>& h_weights, FLB::Mesh* mesh, size_t maxIterations, unsigned int numDimensions, unsigned int numVelocities, Fl_Simple_Terminal* terminal, FLB::RenderLayer* renderLayer, std::filesystem::path& directorySave)
{
  size_t numPointsMesh = mesh -> getNumberPointsMesh();
  // Distribution function
  std::vector<PRECISION> h_f(numVelocities * numPointsMesh, 0);
  PRECISION* d_f; 

  // Velocity
  std::vector<PRECISION> h_u(2 * numPointsMesh, 0);
  PRECISION* d_u; // not used when there are graphics
  
  // Type Cell
  std::vector<uint8_t>& h_flags = mesh -> getDomainFlags();
  uint8_t* d_flags;

  // density
  std::vector<PRECISION> h_rho(numPointsMesh, 0);
  PRECISION* d_rho;
  
  // Variables only used in free surface
  // mass
  std::vector<PRECISION> h_mass;
  PRECISION* d_mass;
  /// excess mass. Variable to Redistribute excess mass from flag conversion (mass conservation)
  std::vector<PRECISION> h_excessMass;
  PRECISION* d_excessMass;
  // fill level of each node
  std::vector<PRECISION> h_phi;
  PRECISION* d_phi;
 
  if (optionsCalc.typeProblem == FLB::TypeProblem::FREE_SURFACE)
  {
    h_mass = std::vector<PRECISION>(numPointsMesh, 0);
    h_excessMass = std::vector<PRECISION>(numPointsMesh, 0);
    h_phi = std::vector<PRECISION>(numPointsMesh, 0);
  }
 
  // Size (number of nodes) of the domain in each direction
  unsigned int h_Nx = mesh -> getNx(); 
  unsigned int h_Ny = mesh -> getNy();
  unsigned int h_N = mesh -> getNumberPointsMesh();

  //set units converter SI units - Lattice units
  FLB::Units unitsConverter;
  double xLengthSI = mesh -> getxMax() - mesh -> getxMin();
  unitsConverter.setConversionParameters(xLengthSI, h_Nx - 1, optionsCalc.referenceVelocitySI, optionsCalc.referenceVelocityLB, optionsCalc.density, 1.0);
  optionsCalc.LBVelocity = unitsConverter.velocitySIToVelocityLB(optionsCalc.SIVelocity);

  // Init data of the domain
  FLB::initData<PRECISION, FLB::initFields<PRECISION>, FLB::initFreeSurfaceFields<PRECISION>>(numVelocities, mesh, h_f, h_u, h_weights, h_rho, h_phi, h_mass, h_excessMass, &optionsCalc);

  // Init Cuda data
  FLB::h_initConstantDataCuda2D<PRECISION>(optionsCalc, h_weights.data(), h_Nx, h_Ny, h_N, unitsConverter);
  FLB::h_initDataCuda2D<PRECISION>(optionsCalc, numPointsMesh, numDimensions, numVelocities, &d_f, h_f.data(), &d_u, h_u.data(), &d_flags, h_flags.data(), &d_mass, h_mass.data(), &d_rho, h_rho.data(), &d_excessMass, h_excessMass.data(), &d_phi, h_phi.data());

  if (optionsCalc.typeProblem == FLB::TypeProblem::MONOFLUID)
  {
    if (optionsCalc.graphicsAPI == FLB::API::OPENGL)
    {
      FLB::h_runCudaMonoFluidOpenGL2D<PRECISION>(optionsCalc, mesh, terminal, renderLayer, unitsConverter, h_u, d_f, d_rho, d_flags, d_mass);
    }
    else if (optionsCalc.graphicsAPI == FLB::API::NONE)
    {
      FLB::h_runCudaMonoFluidNoGraphics2D<PRECISION>(optionsCalc, mesh, terminal, unitsConverter, d_u, h_u, d_f, d_rho, h_rho, d_flags, h_flags, d_mass, directorySave);
    } 
  }

  else if (optionsCalc.typeProblem == FLB::TypeProblem::FREE_SURFACE)
  {
    if (optionsCalc.graphicsAPI == FLB::API::OPENGL)
    {
      FLB::h_runCudaFreeSurfaceOpenGL2D<PRECISION>(optionsCalc, mesh, terminal, renderLayer, unitsConverter, h_u, d_f, d_rho, d_flags, d_mass, d_excessMass, d_phi);
    }
    else if (optionsCalc.graphicsAPI == FLB::API::NONE)
    {
      FLB::h_runCudaFreeSurfaceNoGraphics2D<PRECISION>(optionsCalc, mesh, terminal, unitsConverter, d_u, h_u, d_f, d_rho, h_rho, d_flags, h_flags, d_mass, d_excessMass, d_phi, h_phi, directorySave);
    }
  }

  // cleanup rest of device data
  cudaFree(d_f);
  cudaFree(d_flags);
  cudaFree(d_rho);

  if (optionsCalc.typeProblem == FLB::TypeProblem::FREE_SURFACE)
  {
    cudaFree(d_mass);
    cudaFree(d_excessMass);
    cudaFree(d_phi);
  }

  // if there is an API, the API clean the Variable
  if (optionsCalc.graphicsAPI == FLB::API::NONE)
  {
    cudaFree(d_u);
  }
}

template<typename PRECISION>
void FLB::h_runCudaMonoFluidOpenGL2D(FLB::OptionsCalculation& optionsCalc, FLB::Mesh* mesh, Fl_Simple_Terminal* terminal, FLB::RenderLayer* renderLayer, FLB::Units& unitsConverter, std::vector<PRECISION>& h_u, PRECISION* d_f, PRECISION* d_rho, uint8_t* d_flags, PRECISION* d_mass)
{  
  const int* typeRendering = renderLayer -> getTypeRendering();

  // Initialization of the variables that will be used for the graphics
  // They will be stored as vertex data in the GPU's memory as a vertex buffer object and through a pointer they will be register in CUDA to do the calculations
  std::unique_ptr<FLB::VertexBuffer> vertexBufferU = FLB::VertexBuffer::create(optionsCalc.graphicsAPI, terminal, h_u.data(), 2 * mesh -> getNumberPointsMesh() * sizeof(PRECISION));
  FLB::BufferLayout layout = {
    {ShaderDataType::SOAFloat2, "a_U", false, mesh -> getNumberPointsMesh()}
  };
  vertexBufferU -> setLayout(layout);
  mesh -> addVertexBuffer(vertexBufferU.get());
 
  //Initialize CUDA's variables that they will be used to reference OpenGL's vertex buffer objects
  struct cudaGraphicsResource* cudaVertexBufferU;
  // Register OpenGL's vertex buffer objects with CUDA
  cudaGraphicsGLRegisterBuffer(&cudaVertexBufferU, vertexBufferU -> getVertexBufferID(), cudaGraphicsMapFlagsNone);

  // Register textures in CUDA
  struct cudaGraphicsResource* cudaTextureFields;
  unsigned int flags = cudaGraphicsRegisterFlagsWriteDiscard | cudaGraphicsRegisterFlagsSurfaceLoadStore;
  checkCudaErrors(cudaGraphicsGLRegisterImage(&cudaTextureFields, (GLuint)FLB::Renderer::getScalarFieldsTexture() -> getID(), GL_TEXTURE_2D, flags));

  cudaSurfaceObject_t d_SurfaceTexture = 0; // surface to write to texture

  // get block size and grid size for cuda kernels
  dim3 blockSize(optionsCalc.cudaBlockSize.x, optionsCalc.cudaBlockSize.y);
  dim3 gridSize = FLB::CudaUtils::getGridSize(2, mesh, blockSize);

  renderLayer -> setUsedFreeMemoryGPU(FLB::CudaUtils::getUsedFreeMemory(terminal));

  // variables used to control time
  auto previousTime = std::chrono::high_resolution_clock::now();
  auto previousTimeSim = std::chrono::high_resolution_clock::now();
  auto currentTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> durationGraphics, durationCalculation;
  float secondsFrameRate = renderLayer -> getSecondsFrameRate();
    
  float countSecondsGraphics = 0.0f;
  bool updateGraphics = countSecondsGraphics > secondsFrameRate;
  
  size_t t = 0; // index of the time interval
    
  // variables that will be used as a pointer to access the vertex buffer objects from CUDA code
  PRECISION* cudaU;
  // Map the CUDA graphics resource to a CUDA device pointer
  cudaGraphicsMapResources(1, &cudaVertexBufferU, 0);
  size_t memorySizeU;
  cudaGraphicsResourceGetMappedPointer((void**)&cudaU, &memorySizeU, cudaVertexBufferU);

  while (FLB::App::s_RunningGraphics)
  {
    previousTimeSim = std::chrono::high_resolution_clock::now();
    if (FLB::Renderer::s_UpdateRender)
    {
      // Perform LBM calculation
      FLB::d_StreamCollide2D<PRECISION, 9><<<gridSize, blockSize>>>(d_f, d_rho, cudaU, d_flags, d_mass, t);
      //cudaDeviceSynchronize();
      //if (t >= 5) break;
      t += 1;
      if (t%200 == 0)
      {
	// set stats of the metrics panel
	renderLayer -> setTime(unitsConverter.timeToSIUnits(t));
	renderLayer -> setCalculationTimeStats(durationCalculation.count());
      }

      if (updateGraphics)
      {
	// Point data
	if (*typeRendering == 9) {;}//vPrueba<PRECISION><<<gridSize, blockSize>>>(r0, cudaU, cudaUy);
	else 
	{
	  // Map the CUDA graphics resource of the texture to a CUDA array
	  cudaGraphicsMapResources(1, &cudaTextureFields, 0);
	  cudaArray* texTureCudaArray;
	  cudaGraphicsSubResourceGetMappedArray(&texTureCudaArray, cudaTextureFields, 0, 0);

	  // Specify surface
	  struct cudaResourceDesc resourceDesc;
	  memset(&resourceDesc, 0, sizeof(resourceDesc));
	  resourceDesc.resType = cudaResourceTypeArray;
	  resourceDesc.res.array.array = texTureCudaArray;

	  // TODO Maybe it is better to use a texture instead of a surface
	  cudaCreateSurfaceObject(&d_SurfaceTexture, &resourceDesc);
	  FLB::CudaUtils::save2DDataToOpenGL<PRECISION><<<gridSize, blockSize>>>(cudaU, d_SurfaceTexture);
	  cudaGraphicsUnmapResources(1, &cudaTextureFields);
	}
      }
    }
    else renderLayer -> setCalculationTimeStats(0.0f);
    
    currentTime = std::chrono::high_resolution_clock::now();
    durationCalculation = currentTime - previousTimeSim;

    // only if there is isoSurface. Because the camera can move, the possibility of updating mut be given even if the calculation is stopped
    if (*renderLayer -> isIsosurfaceRendering() && updateGraphics)
    {
      const FLB::IsoSurfaceComponent& isoSurfaceComponent = renderLayer -> getIsoSurfaceComponent();
      // get cuda resource of the texture mapped
      struct cudaGraphicsResource* cudaTextureIsoSurface = isoSurfaceComponent.cudaResTextureIsoSurface;
      // Map the texture of the IsoSurface
      cudaGraphicsMapResources(1, &cudaTextureIsoSurface, 0);
      cudaArray* texTureCudaArrayIso;
      cudaGraphicsSubResourceGetMappedArray(&texTureCudaArrayIso, cudaTextureIsoSurface, 0, 0);
      
      // Specify surface
      struct cudaResourceDesc resourceDesc;
      memset(&resourceDesc, 0, sizeof(resourceDesc));
      resourceDesc.resType = cudaResourceTypeArray;
      resourceDesc.res.array.array = texTureCudaArrayIso;
      
      cudaCreateSurfaceObject(&d_SurfaceTexture, &resourceDesc);

      // get bounds of the lattice nodes visible by the camera
      unsigned int cameraBounds[4];
      renderLayer -> getCameraDomainBounds(cameraBounds);

      switch (isoSurfaceComponent.type) 
      {
	case 0: // velocity
	{
	  FLB::d_CreateIsoSurfaceVector2D<PRECISION><<<gridSize, blockSize>>>(cudaU, d_SurfaceTexture, isoSurfaceComponent.textureWidth, isoSurfaceComponent.textureHeight, isoSurfaceComponent.isoValue, cameraBounds[0], cameraBounds[1], cameraBounds[2], cameraBounds[3]);
	  break;
	}
	//cudaDeviceSynchronize();
      }
      cudaGraphicsUnmapResources(1, &cudaTextureIsoSurface);
    }
    cudaDeviceSynchronize();
    //checkCudaErrors(cudaGetLastError());
 
    durationGraphics = currentTime - previousTime;
    countSecondsGraphics = durationGraphics.count();

    updateGraphics = countSecondsGraphics > secondsFrameRate;
     
    if (updateGraphics)
    {
      // Unmap the vertex buffer object so that OpenGL can access the resource
      cudaGraphicsUnmapResources(1, &cudaVertexBufferU);

      renderLayer -> onUpdate();
      secondsFrameRate = renderLayer -> getSecondsFrameRate(); // update this value because it can change if the user decide it
      previousTime = currentTime;

      // Map the CUDA graphics resource to a CUDA device pointer
      cudaGraphicsMapResources(1, &cudaVertexBufferU, 0);
      size_t memorySizeU;
      cudaGraphicsResourceGetMappedPointer((void**)&cudaU, &memorySizeU, cudaVertexBufferU);
    }
  }
  //cleanup graphics resources used
  cudaGraphicsUnregisterResource(cudaVertexBufferU);
  cudaGraphicsUnregisterResource(cudaTextureFields);
  cudaDestroySurfaceObject(d_SurfaceTexture);
}

template<typename PRECISION> 
void FLB::h_runCudaFreeSurfaceOpenGL2D(FLB::OptionsCalculation& optionsCalc, FLB::Mesh* mesh, Fl_Simple_Terminal* terminal, FLB::RenderLayer* renderLayer, FLB::Units& unitsConverter, std::vector<PRECISION>& h_u, PRECISION* d_f, PRECISION* d_rho, uint8_t* d_flags, PRECISION* d_mass, PRECISION* d_excessMass, PRECISION* d_phi)
{
  const int* typeRendering = renderLayer -> getTypeRendering();
  
  // Initialization of the variables that will be used for the graphics
  // They will be stored as vertex data in the GPU's memory as a vertex buffer object and through a pointer they will be register in CUDA to do the calculations
  std::unique_ptr<FLB::VertexBuffer> vertexBufferU = FLB::VertexBuffer::create(optionsCalc.graphicsAPI, terminal, h_u.data(), 2 * mesh -> getNumberPointsMesh() * sizeof(PRECISION));
  FLB::BufferLayout layout = {
    {ShaderDataType::SOAFloat2, "a_U", false, mesh -> getNumberPointsMesh()}
  };
  vertexBufferU -> setLayout(layout);
  mesh -> addVertexBuffer(vertexBufferU.get());
  
  //Initialize CUDA's variables that they will be used to reference OpenGL's vertex buffer objects
  struct cudaGraphicsResource* cudaVertexBufferU;
  // Register OpenGL's vertex buffer objects with CUDA
  cudaGraphicsGLRegisterBuffer(&cudaVertexBufferU, vertexBufferU -> getVertexBufferID(), cudaGraphicsMapFlagsNone);

  // Register textures in CUDA
  struct cudaGraphicsResource* cudaResourceTexture;
  checkCudaErrors(cudaGraphicsGLRegisterImage(&cudaResourceTexture, (GLuint)FLB::Renderer::getScalarFieldsTexture() -> getID(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
  
  cudaSurfaceObject_t d_SurfaceTexture = 0;

  dim3 blockSize(optionsCalc.cudaBlockSize.x, optionsCalc.cudaBlockSize.y);
  dim3 gridSize = FLB::CudaUtils::getGridSize(2, mesh, blockSize);

  renderLayer -> setUsedFreeMemoryGPU(FLB::CudaUtils::getUsedFreeMemory(terminal));
  
  // variables used to control time
  auto previousTime = std::chrono::high_resolution_clock::now();
  auto previousTimeSim = std::chrono::high_resolution_clock::now();
  auto currentTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> durationGraphics, durationCalculation;
  float secondsFrameRate = renderLayer -> getSecondsFrameRate();
    
  float countSecondsGraphics = 0.0f;
  bool updateGraphics = countSecondsGraphics > secondsFrameRate;
  
  size_t t = 0; // index of the time interval
  
  // variables that will be used as a pointer to access the vertex buffer objects from CUDA code
  PRECISION* cudaU;
  // Map the CUDA graphics resource to a CUDA device pointer
  cudaGraphicsMapResources(1, &cudaVertexBufferU, 0);
  size_t memorySizeU;
  cudaGraphicsResourceGetMappedPointer((void**)&cudaU, &memorySizeU, cudaVertexBufferU);
 
  while (FLB::App::s_RunningGraphics)
  {
    previousTimeSim = std::chrono::high_resolution_clock::now();
    if (FLB::Renderer::s_UpdateRender)
    {
      // Perform LBM calculation
      FLB::d_FreeSurface2D_1<PRECISION, 9><<<gridSize, blockSize>>>(d_f, d_rho, cudaU, d_flags, d_mass, d_excessMass, d_phi, t);
      std::cout << "FREE_1\n";
      FLB::d_StreamCollide2D<PRECISION, 9><<<gridSize, blockSize>>>(d_f, d_rho, cudaU, d_flags, d_mass, t);
      FLB::d_FreeSurface2D_2<9><<<gridSize, blockSize>>>(d_flags);
      std::cout << "FREE_2\n";
      FLB::d_FreeSurface2D_3<PRECISION, 9><<<gridSize, blockSize>>>(d_f, d_rho, cudaU, d_flags, t);
      std::cout << "FREE_3\n";
      FLB::d_FreeSurface2D_4<PRECISION, 9><<<gridSize, blockSize>>>(d_rho, d_flags, d_mass, d_excessMass, d_phi);
      std::cout << "FREE_4\n";
      std::cout <<"t = " << t<< "\n\n";
      if (t >= 10000000) FLB::Renderer::s_UpdateRender = false;
      t += 1;
      if (t%200 == 0)
      {
	// set stats of the metrics panel
	renderLayer -> setTime(unitsConverter.timeToSIUnits(t));
	renderLayer -> setCalculationTimeStats(durationCalculation.count());
      }

      if (updateGraphics)
      {
	// Point data
	if(*typeRendering == 9) {;}//vPrueba<PRECISION><<<gridSize, blockSize>>>(r0, cudaU, cudaUy);
	else 
	{
	  // Map the CUDA graphics resource of the texture to a CUDA array
	  cudaGraphicsMapResources(1, &cudaResourceTexture, 0);
	  cudaArray* texTureCudaArray;
	  cudaGraphicsSubResourceGetMappedArray(&texTureCudaArray, cudaResourceTexture, 0, 0);

	  struct cudaResourceDesc resourceDesc;
	  memset(&resourceDesc, 0, sizeof(resourceDesc));

	  resourceDesc.resType = cudaResourceTypeArray;
	  resourceDesc.res.array.array = texTureCudaArray;

	  cudaCreateSurfaceObject(&d_SurfaceTexture, &resourceDesc);
	  FLB::CudaUtils::save2DDataToOpenGL<PRECISION><<<gridSize, blockSize>>>(cudaU, d_SurfaceTexture);
	  cudaGraphicsUnmapResources(1, &cudaResourceTexture);
	}
      }
    }
    else renderLayer -> setCalculationTimeStats(0.0f);
    
    currentTime = std::chrono::high_resolution_clock::now();
    durationCalculation = currentTime - previousTimeSim;
    
    // only if there is isoSurface. Because the camera can move, the possibility of updating mut be given even if the calculation is stopped
    if (*renderLayer -> isIsosurfaceRendering() && updateGraphics)
    {
      const FLB::IsoSurfaceComponent& isoSurfaceComponent = renderLayer -> getIsoSurfaceComponent();
      // get cuda resource of the texture mapped
      struct cudaGraphicsResource* cudaTextureIsoSurface = isoSurfaceComponent.cudaResTextureIsoSurface;
      // Map the texture of the IsoSurface
      cudaGraphicsMapResources(1, &cudaTextureIsoSurface, 0);
      cudaArray* texTureCudaArrayIso;
      cudaGraphicsSubResourceGetMappedArray(&texTureCudaArrayIso, cudaTextureIsoSurface, 0, 0);
      
      // Specify surface
      struct cudaResourceDesc resourceDesc;
      memset(&resourceDesc, 0, sizeof(resourceDesc));
      resourceDesc.resType = cudaResourceTypeArray;
      resourceDesc.res.array.array = texTureCudaArrayIso;
      
      cudaCreateSurfaceObject(&d_SurfaceTexture, &resourceDesc);

      // get bounds of the lattice nodes visible by the camera
      unsigned int cameraBounds[4];
      renderLayer -> getCameraDomainBounds(cameraBounds);

      switch (isoSurfaceComponent.type) 
      {
	case 0: // velocity
	{
	  FLB::d_CreateIsoSurfaceVector2D<PRECISION><<<gridSize, blockSize>>>(cudaU, d_SurfaceTexture, isoSurfaceComponent.textureWidth, isoSurfaceComponent.textureHeight, isoSurfaceComponent.isoValue, cameraBounds[0], cameraBounds[1], cameraBounds[2], cameraBounds[3]);
	  break;
	}
	case 1: // phi
	{
	  FLB::d_CreateIsoSurfaceVector2D<PRECISION><<<gridSize, blockSize>>>(d_phi, d_SurfaceTexture, isoSurfaceComponent.textureWidth, isoSurfaceComponent.textureHeight, 0.5f, cameraBounds[0], cameraBounds[1], cameraBounds[2], cameraBounds[3]);
	  break;
	}
	//cudaDeviceSynchronize();
      }
      cudaGraphicsUnmapResources(1, &cudaTextureIsoSurface);

    }
    cudaDeviceSynchronize();
    //checkCudaErrors(cudaGetLastError());

    durationGraphics = currentTime - previousTime;
    countSecondsGraphics = durationGraphics.count();
    updateGraphics = countSecondsGraphics > secondsFrameRate;
    
    if (updateGraphics)
    {
      // Unmap the vertex buffer object so that OpenGL can access the resource
      cudaGraphicsUnmapResources(1, &cudaVertexBufferU);	 
      
      renderLayer -> onUpdate();
      secondsFrameRate = renderLayer -> getSecondsFrameRate(); // update this value because it can change if the user decide it
      previousTime = currentTime;

      // Map the CUDA graphics resource to a CUDA device pointer
      cudaGraphicsMapResources(1, &cudaVertexBufferU, 0);
      size_t memorySizeU;
      cudaGraphicsResourceGetMappedPointer((void**)&cudaU, &memorySizeU, cudaVertexBufferU);
    }  
  }
  //cleanup graphics resources used
  cudaGraphicsUnregisterResource(cudaVertexBufferU);
  cudaGraphicsUnregisterResource(cudaResourceTexture);
  cudaDestroySurfaceObject(d_SurfaceTexture);
}

template<typename PRECISION>
void FLB::h_runCudaMonoFluidNoGraphics2D(FLB::OptionsCalculation& optionsCalc, FLB::Mesh* mesh, Fl_Simple_Terminal* terminal, FLB::Units& unitsConverter, PRECISION* d_u, std::vector<PRECISION>& h_u, PRECISION* d_f, PRECISION* d_rho, std::vector<PRECISION>& h_rho, uint8_t* d_flags, std::vector<uint8_t>& h_flags, PRECISION* d_mass, std::filesystem::path& directorySave)
{
  size_t t = 0; // index of the time interval
  size_t timeSimulation = static_cast<size_t>(unitsConverter.timeToLatticeUnits(optionsCalc.timeSimulation));
  size_t timeSave = static_cast<size_t>(unitsConverter.timeToLatticeUnits(optionsCalc.timeSave));
  
  dim3 blockSize(optionsCalc.cudaBlockSize.x, optionsCalc.cudaBlockSize.y);
  dim3 gridSize = FLB::CudaUtils::getGridSize(2, mesh, blockSize);
  
  FLB::VTIWriter writer{directorySave};
  std::string typeFloatDataVTK = std::is_same_v<PRECISION, float> ? "Float32" : "Float64";

  while (t <= timeSimulation)
  {
    // Perform LBM calculation
    FLB::d_StreamCollide2D<PRECISION, 9><<<gridSize, blockSize>>>(d_f, d_rho, d_u, d_flags, d_mass, t);
    cudaDeviceSynchronize();
    if (t%1000 == 0)std::cout <<"t = "<< unitsConverter.timeToSIUnits(t) << "\n";

    // save results
    if (t % timeSave == 0)
    {
      FLB::CudaUtils::copyDataFromDevice<PRECISION>(mesh -> getNumberPointsMesh(), {{2, sizeof(PRECISION), h_u.data(), d_u}, {1, sizeof(PRECISION), h_rho.data(), d_rho}}, h_flags.data(), d_flags);
      writer.addField<uint8_t>("UInt8", "Flags", h_flags, mesh -> getNumberPointsMesh(), true);
      writer.addField<PRECISION>(typeFloatDataVTK, "U", h_u, mesh -> getNumberPointsMesh(), false, unitsConverter.getVelocityParameterToSIUnits(), 2, false); // In cuda the y axis is downwards, so it is necessary to change the sign
      writer.addField<PRECISION>(typeFloatDataVTK, "rho", h_rho, mesh -> getNumberPointsMesh(), true, unitsConverter.getRhoParameterToSIUnits());
      writer.writeData(mesh, terminal, false, optionsCalc.timeSave);
    }
    t += 1;
  }
}

template<typename PRECISION>
void FLB::h_runCudaFreeSurfaceNoGraphics2D(FLB::OptionsCalculation &optionsCalc, FLB::Mesh *mesh, Fl_Simple_Terminal *terminal, FLB::Units& unitsConverter,  PRECISION* d_u, std::vector<PRECISION> &h_u, PRECISION *d_f, PRECISION *d_rho, std::vector<PRECISION>& h_rho, uint8_t *d_flags, std::vector<uint8_t>& h_flags, PRECISION *d_mass, PRECISION *d_excessMass, PRECISION *d_phi, std::vector<PRECISION>& h_phi, std::filesystem::path& directorySave)
{
  size_t t = 0; // index of the time interval
  size_t timeSimulation = static_cast<size_t>(unitsConverter.timeToLatticeUnits(optionsCalc.timeSimulation));
  size_t timeSave = static_cast<size_t>(unitsConverter.timeToLatticeUnits(optionsCalc.timeSave));
  
  dim3 blockSize(optionsCalc.cudaBlockSize.x, optionsCalc.cudaBlockSize.y);
  dim3 gridSize = FLB::CudaUtils::getGridSize(2, mesh, blockSize);
  
  FLB::VTIWriter writer{directorySave};
  std::string typeFloatDataVTK = std::is_same_v<PRECISION, float> ? "Float32" : "Float64";
  
  while (t <= timeSimulation)
  {
    // Perform LBM calculation
    FLB::d_FreeSurface2D_1<PRECISION, 9><<<gridSize, blockSize>>>(d_f, d_rho, d_u, d_flags, d_mass, d_excessMass, d_phi, t);
    FLB::d_StreamCollide2D<PRECISION, 9><<<gridSize, blockSize>>>(d_f, d_rho, d_u, d_flags, d_mass, t);
    FLB::d_FreeSurface2D_2<9><<<gridSize, blockSize>>>(d_flags);
    FLB::d_FreeSurface2D_3<PRECISION, 9><<<gridSize, blockSize>>>(d_f, d_rho, d_u, d_flags, t);
    FLB::d_FreeSurface2D_4<PRECISION, 9><<<gridSize, blockSize>>>(d_rho, d_flags, d_mass, d_excessMass, d_phi);
    
    // save results
    if (t % timeSave == 0)
    {
      FLB::CudaUtils::copyDataFromDevice<PRECISION>(mesh -> getNumberPointsMesh(), {{2, sizeof(PRECISION), h_u.data(), d_u}, {1, sizeof(PRECISION), h_rho.data(), d_rho}, {1, sizeof(PRECISION), h_phi.data(), d_phi}}, h_flags.data(), d_flags);
      writer.addField<uint8_t>("UInt8", "Flags", h_flags, mesh -> getNumberPointsMesh(), true);
      writer.addField<PRECISION>(typeFloatDataVTK, "U", h_u, mesh -> getNumberPointsMesh(), false, 1.0, 2, false); // In cuda the y axis is downwards, so it is necessary to change the sign
      writer.addField<PRECISION>(typeFloatDataVTK, "rho", h_rho, mesh -> getNumberPointsMesh(), true);
      writer.addField<PRECISION>(typeFloatDataVTK, "phi", h_phi, mesh -> getNumberPointsMesh(), true);
      writer.writeData(mesh, terminal, false, optionsCalc.timeSave);
    }

    t += 1;
  }
}

// Explicit instantation of the functions because the templates functions are instantiated in a diferent compilation unit
template void FLB::h_launchCudaCalculations2D<float>(FLB::OptionsCalculation& optionsCalc, std::vector<float>& h_weights, FLB::Mesh* mesh, size_t maxIterations, unsigned int numDimensions, unsigned int numVelocities, Fl_Simple_Terminal* terminal, FLB::RenderLayer* renderLayer, std::filesystem::path& pathSave);

template void FLB::h_launchCudaCalculations2D<double>(FLB::OptionsCalculation& optionsCalc, std::vector<double>& h_weights, FLB::Mesh* mesh, size_t maxIterations, unsigned int numDimensions, unsigned int numVelocities, Fl_Simple_Terminal* terminal, FLB::RenderLayer* renderLayer, std::filesystem::path& pathSave);
