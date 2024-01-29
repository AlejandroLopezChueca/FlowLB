#include "runCuda.cuh"
#include "cudaInitData.cuh"
#include "lbmKernel2D.cuh"
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

  std::vector<PRECISION> h_rho(numPointsMesh, 0);
  PRECISION* d_rho;
  
  // mass. It is used only for free surface
  std::vector<PRECISION> h_mass;
  PRECISION* d_mass;
  
  if (optionsCalc.typeProblem == FLB::TypeProblem::FREE_SURFACE)
  {
    h_mass = std::vector<PRECISION>(numPointsMesh, 0);
  }
 
  // Size (number of nodes) of the domain in each direction
  unsigned int h_Nx = mesh -> getNx(); 
  unsigned int h_Ny = mesh -> getNy();
  unsigned int h_N = mesh -> getNumberPointsMesh();

  uint8_t h_collisionOperator = optionsCalc.collisionOperator;

  //set units converter SI units - Lattice units
  FLB::Units unitsConverter;
  double xLengthSI = mesh -> getxMax() - mesh -> getxMin();
  unitsConverter.setConversionParameters(xLengthSI, h_Nx - 1, optionsCalc.velocity, optionsCalc.LBVelocity, optionsCalc.density, 1.0);

  float h_g = unitsConverter.gToLatticeUnits(optionsCalc.gravity);
  float h_nu = unitsConverter.nuToLatticeUnits(optionsCalc.kinematicViscosity);

  // Init data of the domain
  FLB::initData<PRECISION, FLB::initFields<PRECISION>>(numVelocities, mesh, h_f, h_u, h_weights, h_rho, &optionsCalc);

  // Init Cuda data
  FLB::h_initConstantDataCuda2D<PRECISION>(optionsCalc, h_weights.data(), h_Nx, h_Ny, h_N, h_collisionOperator, h_g, h_nu);
  FLB::h_initDataCuda2D<PRECISION>(optionsCalc, numPointsMesh, numDimensions, numVelocities, &d_f, h_f.data(), &d_u, h_u.data(), &d_flags, h_flags.data(), &d_mass, h_mass.data(), &d_rho, h_rho.data());

  if (optionsCalc.typeProblem == FLB::TypeProblem::MONOFLUID && optionsCalc.graphicsAPI == FLB::API::OPENGL)
  {
    FLB::h_runCudaMonoFluidOpenGL2D<PRECISION>(optionsCalc, mesh, terminal, renderLayer, unitsConverter, h_u, d_f, d_rho, d_flags, d_mass);
  }

  else if (optionsCalc.typeProblem == FLB::TypeProblem::MONOFLUID && optionsCalc.graphicsAPI == FLB::API::NONE)
  {
    FLB::h_runCudaMonoFluidNoGraphics2D<PRECISION>(optionsCalc, mesh, terminal, unitsConverter, d_u, h_u, d_f, d_rho, d_flags, d_mass, directorySave);
  }

  // cleanup rest of device data
  cudaFree(d_f);
  cudaFree(d_flags);
  cudaFree(d_mass);
  cudaFree(d_rho);

  if (optionsCalc.graphicsAPI == FLB::API::NONE)
  {
    cudaFree(d_u);
  }
}

template<typename PRECISION>
void FLB::h_runCudaMonoFluidOpenGL2D(FLB::OptionsCalculation& optionsCalc, FLB::Mesh* mesh, Fl_Simple_Terminal* terminal, FLB::RenderLayer* renderLayer, FLB::Units& unitsConverter, std::vector<PRECISION>& h_u, PRECISION* d_f, PRECISION* d_rho, uint8_t* d_flags, PRECISION* d_mass)
{
  float prevTime = 0.0f;
  float currentTime = 0.0f;
  float timeDiff;
  unsigned int counterFrames = 0;
  
  size_t t = 0; // index of the time interval
  
  const int* typeRendering = renderLayer -> getTypeRendering();

  // Initialization of the variables that will be used for the graphics
  // They will be stored as vertex data in the GPU's memory as a vertex buffer object
  // and through a pointer they will be register in CUDA to do the calculations
  std::unique_ptr<FLB::VertexBuffer> vertexBufferU = FLB::VertexBuffer::create(optionsCalc.graphicsAPI, terminal, h_u.data(), 2 * mesh -> getNumberPointsMesh() * sizeof(PRECISION));
  FLB::BufferLayout layout = {
    {ShaderDataType::SOAFloat2, "a_U", false, mesh -> getNumberPointsMesh()}
  };
  vertexBufferU -> setLayout(layout);
  mesh -> addVertexBuffer(vertexBufferU.get());
 
  //Initialize CUDA's variables that they will be used to reference OpenGL's vertex buffer objects
  struct cudaGraphicsResource* cudaVertexBufferU;
  //struct cudaGraphicsResource* cudaVertexBufferUy;
  // Register OpenGL's vertex buffer objects with CUDA
  cudaGraphicsGLRegisterBuffer(&cudaVertexBufferU, vertexBufferU -> getVertexBufferID(), cudaGraphicsMapFlagsNone);

  // Register textures in CUDA
  struct cudaGraphicsResource* cudaResourceTexture;
  checkCudaErrors(cudaGraphicsGLRegisterImage(&cudaResourceTexture, (GLuint)FLB::Renderer::getScalarFieldsTexture() -> getID(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));

  cudaSurfaceObject_t d_SurfaceTexture = 0;

  dim3 blockSize(optionsCalc.cudaBlockSize.x, optionsCalc.cudaBlockSize.y);
  dim3 gridSize = FLB::CudaUtils::getGridSize(2, mesh, blockSize);

  //std::cout << gridSize.x<< " " << gridSize.y << " BL\n";
  //std::cout << mesh -> getNx()<< " " << mesh->getNy() << " BL\n";
  
  while (FLB::App::s_RunningGraphics)
  {
    if (FLB::Renderer::s_UpdateRender)
    {
      // variables that will be used as a pointer to access the vertex buffer objects from CUDA code
      PRECISION* cudaU;
      //PRECISION* cudaUy;
      // Map the CUDA graphics resource to a CUDA device pointer
      cudaGraphicsMapResources(1, &cudaVertexBufferU, 0);
      size_t memorySizeU;
      cudaGraphicsResourceGetMappedPointer((void**)&cudaU, &memorySizeU, cudaVertexBufferU);

      // Perform LBM calculation
      FLB::d_StreamCollide2D<PRECISION, 9><<<gridSize, blockSize>>>(d_f, d_rho, cudaU, d_flags, d_mass, t);
      cudaDeviceSynchronize();
      //if (t >= 5) break;
      t += 1;
      if (t%1000 == 0)
      {
	//std::cout <<"t = "<< unitsConverter.timeToSIUnits(t) << "\n";
	renderLayer -> setTime(unitsConverter.timeToSIUnits(t));
      }

      //TODO TEMPORAL
      float r0;
      r0 = ((double)(rand())/RAND_MAX);

      //TODO TEMPORAL
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
      //checkCudaErrors(cudaGetLastError());

      // Unmap the vertex buffer object so that OpenGL can access the resource
      cudaGraphicsUnmapResources(1, &cudaVertexBufferU);	 

    }
    renderLayer -> onUpdate();
  }
  //cleanup graphics resources used
  cudaGraphicsUnregisterResource(cudaVertexBufferU);
  cudaGraphicsUnregisterResource(cudaResourceTexture);
  cudaDestroySurfaceObject(d_SurfaceTexture);
}

template<typename PRECISION>
void FLB::h_runCudaMonoFluidNoGraphics2D(FLB::OptionsCalculation& optionsCalc, FLB::Mesh* mesh, Fl_Simple_Terminal* terminal, FLB::Units& unitsConverter, PRECISION* d_u, std::vector<PRECISION>& h_u, PRECISION* d_f, PRECISION* d_rho, uint8_t* d_flags, PRECISION* d_mass, std::filesystem::path& directorySave)
{
  size_t t = 0; // index of the time interval
  size_t timeSimulation = static_cast<size_t>(unitsConverter.timeToLatticeUnits(optionsCalc.timeSimulation));
  size_t timeSave = static_cast<size_t>(unitsConverter.timeToLatticeUnits(optionsCalc.timeSave));
  
  dim3 blockSize(optionsCalc.cudaBlockSize.x, optionsCalc.cudaBlockSize.y);
  dim3 gridSize = FLB::CudaUtils::getGridSize(2, mesh, blockSize);
  
  FLB::VTIWriter writer{directorySave};
  std::string typeFloatDataVTK = std::is_same_v<float, float> ? "Float32" : "Float64";

  while (t <= timeSimulation)
  {
    // Perform LBM calculation
    FLB::d_StreamCollide2D<PRECISION, 9><<<gridSize, blockSize>>>(d_f, d_rho, d_u, d_flags, d_mass, t);
    cudaDeviceSynchronize();
    if (t%1000 == 0)std::cout <<"t = "<< unitsConverter.timeToSIUnits(t) << "\n";

    // save results
    if (t % timeSave == 0)
    {
      FLB::CudaUtils::copyDataFromDevice<PRECISION>(mesh -> getNumberPointsMesh(), 2, d_u, h_u.data());
      writer.addField<uint8_t>("UInt8", "Flags", mesh -> getDomainFlags(), mesh -> getNumberPointsMesh(), true);
      writer.addField<PRECISION>(typeFloatDataVTK, "U", h_u, mesh -> getNumberPointsMesh(), false, 2);
      writer.writeData(mesh, terminal, false, optionsCalc.timeSave);
    }
    //if (t % 100) terminal -> printf(" TIME : %6.4f seg\n", unitsConverter.timeToSIUnits(t));

    t += 1;
  }
}



// Explicit instantation of the functions because the templates functions are instantiated in a diferent compilation unit
template void FLB::h_launchCudaCalculations2D<float>(FLB::OptionsCalculation& optionsCalc, std::vector<float>& h_weights, FLB::Mesh* mesh, size_t maxIterations, unsigned int numDimensions, unsigned int numVelocities, Fl_Simple_Terminal* terminal, FLB::RenderLayer* renderLayer, std::filesystem::path& pathSave);


//template void FLB::h_runCudaCalculations2D<double>(FLB::OptionsCalculation& optionsCalc, FLB::Window& renderWindow, FLB::VertexArray& vertexArray, FLB::OrthographicCamera& camera, glm::vec3 cameraPosition, double* h_weights, size_t numPointsMesh, size_t maxIterations, unsigned int numDimensions, unsigned int numVelocities);


