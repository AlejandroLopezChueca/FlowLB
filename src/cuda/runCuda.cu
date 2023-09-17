#include "lbmKernel2D.cuh"
#include "runCuda.cuh"
#include "cudaInitData.cuh"
#include "cudaUtils.cuh"

#include "io/writer.h"
#include "initData.h"
#include "graphics/OpenGL/OpenGLWindow.h"
#include "graphics/renderer.h"
#include "graphics/texture.h"
#include <GL/gl.h>
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


template<typename PRECISION>
__global__ void vPrueba(float randNUmber, PRECISION* vx)
{
  //printf("%6.4f\n", randNUmber);
  const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int y = threadIdx.y + blockIdx.y *blockDim.y;
  //const unsigned int idx = x + y * FLB::d_Nx;
  const unsigned int idx = x + y * x;

  //printf("x = %d, y = %d\n", x, y);
  //printf("d_Nx = %d, d_Ny = %d\n", FLB::d_Nx, FLB::d_Ny);
  //if (x >= FLB::d_Nx || y >= FLB::d_Ny) return;
  if (idx > FLB::d_N) return;
  //printf("BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB\n");
  vx[idx] = randNUmber;
  //vx[0] = randNUmber/2;
  //vx[5] = randNUmber/4;
  vx[1] = randNUmber/1.3;
  vx[0] = 0;
  vx[8] = 1;
  vx[18] = 1;
  vx[17] = 1;
  vx[16] = 1;
  vx[22] = 1;
  vx[26] = randNUmber*2;
  vx[12] = randNUmber/1.4;
  //printf("%f\n",idx);
  //printf("%6.4f\n", randNUmber);
  //vy[idx] = curand_uniform(&localState);;
}

template<typename PRECISION>
__global__ void vPrueba2(float randNUmber, PRECISION* vx, cudaSurfaceObject_t d_SurfaceTexture)
{
  const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int y = threadIdx.y + blockIdx.y *blockDim.y;
  const unsigned int idx = x + y * FLB::d_Nx;
  //printf("x = %d, y = %d\n", x, y);
  //printf("d_Nx = %d, d_Ny = %d\n", FLB::d_Nx, FLB::d_Ny);
  if (x >= FLB::d_Nx || y >= FLB::d_Ny) return;
  //if (idx > FLB::d_N) return;
  vx[idx] = randNUmber;
  //vx[0] = randNUmber/2;
  //vx[5] = randNUmber/4;
  vx[1] = randNUmber/1.3;
  vx[0] = 0;
  vx[8] = 1;
  vx[18] = 1;
  vx[17] = 1;
  vx[16] = 1;
  vx[22] = 1;
  vx[26] = randNUmber*2;
  vx[12] = randNUmber/1.4;
  //float4 data = make_float4(vx[idx], vx[8], vx[2],  1.0);
  uchar4 data = make_uchar4(x%255 + randNUmber*200, y%255 + randNUmber *150, randNUmber*255, 255);

  //printf("%6.4f, %6.4f, %6.4f, %6.4f\n", data.x, data.y, data.z, data.w);
  //printf("%d, %d, %d, %d\n", data.x, data.y, data.z, data.w);

  surf2Dwrite(data, d_SurfaceTexture, x*sizeof(uchar4), y);

  uchar4 data2;

  //surf2Dread(&data2, d_SurfaceTexture, x*4,y);
  printf("x = %d, y = %d\n", x, y);
  printf("%d, %d, %d, %d\n", data.x, data.y, data.z, data.w);
  //printf("%d, %d, %d, %d\n\n", data2.x, data2.y, data2.z, data2.w);

  //printf("%f\n",randNUmber);
  //printf("%6.4f\n", randNUmber);
  //vy[idx] = curand_uniform(&localState);;
}


template<typename PRECISION>
void FLB::h_runCudaCalculations2D(FLB::OptionsCalculation& optionsCalc, FLB::Window& renderWindow, FLB::VertexArray& vertexArray, FLB::Shader& textureShader, FLB::OrthographicCameraController& orthographicCameraController, PRECISION* h_weights, FLB::Mesh* mesh, size_t maxIterations, unsigned int numDimensions, unsigned int numVelocities, Fl_Simple_Terminal* terminal)
{
  size_t numPointsMesh = mesh -> numPointsMesh;
  // Distribution function
  std::vector<PRECISION> h_f(numVelocities * numPointsMesh, 0);
  PRECISION* d_f;  
  // Velocity
  std::vector<PRECISION> h_ux(numPointsMesh, 0);
  PRECISION* d_ux;
  std::vector<PRECISION> h_uy(numPointsMesh, 0);
  PRECISION* d_uy;
  // Type Cell
  std::vector<uint8_t> h_flags(numPointsMesh);
  uint8_t* d_flags;
  // mass
  std::vector<PRECISION> h_mass(numPointsMesh, 0);
  PRECISION* d_mass;

  PRECISION* h_rho;
  PRECISION* d_rho;
 

  // Size (number of nodes) of the domain in each direction
  unsigned int h_Nx = mesh -> Nx; 
  unsigned int h_Ny = mesh -> Ny;
  unsigned int h_N = mesh -> numPointsMesh;

  float h_g, h_nu; // TEMPORAL
  uint8_t h_collisionOperator;

  // Init data of the domain
 //FLB::initData<PRECISION, FLB::initVelocityAndf<PRECISION>>(numVelocities, mesh, h_f, h_ux, h_uy, h_weights);
 
  FLB::h_initConstantDataCuda2D<PRECISION>(optionsCalc, h_weights, h_Nx, h_Ny, h_N, h_collisionOperator, h_g, h_nu);
  FLB::h_initDataCuda2D<PRECISION>(optionsCalc, numPointsMesh, numDimensions, numVelocities, d_f, h_f.data(), d_ux, h_ux.data(), d_uy, h_uy.data(), d_flags, h_flags.data(), d_mass, h_mass.data());


  size_t t = 0;

    //d_streamCollide2D<PRECISION, 9> <<<1,1>>>(d_f, d_rho, d_ux, d_uy, d_flags, d_mass, t);

    //if (iteration % OptionsCalc.intervalSave == 0)
    //{
    //  FLB::copyDataFromDevice<PRECISION>(numPointsMesh, numDimensions, numVelocities, d_vx, h_vx);


    //  ++iteration;
    //}
  if (optionsCalc.typeProblem == FLB::TypeProblem::MONOFLUID && optionsCalc.graphicsAPI == FLB::API::OPENGL)
  {
    float prevTime = 0.0f;
    float currentTime = 0.0f;
    float timeDiff;
    unsigned int counterFrames = 0;


    // Initialization of the variables that will be used for the graphics
    // The will be stored as vertex data in the GPU's memory as a vertex buffer object
    // and through a pointer they will be register in CUDA to do the calculations
    std::unique_ptr<FLB::VertexBuffer> vertexBufferUx = FLB::VertexBuffer::create(optionsCalc.graphicsAPI, terminal, h_ux.data(), numPointsMesh * sizeof(PRECISION));
  FLB::BufferLayout layout = {
      {ShaderDataType::Float, "a_Ux"}
    };
    vertexBufferUx -> setLayout(layout);
    vertexArray.addVertexBuffer(vertexBufferUx.get());
    
    std::unique_ptr<FLB::VertexBuffer> vertexBufferUy = FLB::VertexBuffer::create(optionsCalc.graphicsAPI, terminal, h_uy.data(), numPointsMesh * sizeof(PRECISION));
    layout = {
      {ShaderDataType::Float, "a_Uy"}
    };
    vertexBufferUy -> setLayout(layout);
    vertexArray.addVertexBuffer(vertexBufferUy.get());

    //Initialize a CUDA variable that it will be used to reference OpenGL's vertex buffer object
    struct cudaGraphicsResource* cudaVertexBufferUx;
    // Register OpenGL's vertex buffer object with CUDA
    cudaGraphicsGLRegisterBuffer(&cudaVertexBufferUx, vertexBufferUx -> getVertexBufferID(), cudaGraphicsMapFlagsNone);



    //std::unique_ptr<FLB::Texture2D> a = FLB::Texture2D::create(optionsCalc.graphicsAPI, 1600, 900);

    struct cudaGraphicsResource* cudaTexture;
    //checkCudaErrors(cudaGraphicsGLRegisterImage(&cudaTexture, a -> getID(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));

    checkCudaErrors(cudaGraphicsGLRegisterImage(&cudaTexture, (GLuint)FLB::Renderer::s_TextureToUse -> getTextureColorID(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));


    cudaSurfaceObject_t d_SurfaceTexture = 0;

    dim3 threadsPerBlock(16,16);
    dim3 numBlocks((mesh -> Nx + threadsPerBlock.x - 1)/threadsPerBlock.x, (mesh -> Ny + threadsPerBlock.y - 1)/threadsPerBlock.y);

    GLFWwindow* window = (GLFWwindow*)(renderWindow.getWindow());
    while (!glfwWindowShouldClose(window))
    {
      //Update counter frames and time
      /*currentTime = glfwGetTime();
      timeDiff = currentTime - prevTime;
      counterFrames++;
      if (timeDiff >= 1.0f/30.0f)
      {	
	// Creates new title
	std::string FPS = std::to_string((1.0 / timeDiff) * counterFrames);
	std::string ms = std::to_string((timeDiff / counterFrames) * 1000);
	std::string newTitle = "FLowLB 0.1 - " + FPS + "FPS / " + ms + "ms";
	glfwSetWindowTitle(window, newTitle.c_str());

	// Resets times and counter
	prevTime = currentTime;
	counterFrames = 0;
      }*/


      orthographicCameraController.GLFWUpdate(window);
      FLB::Renderer::setClearColor({0.5f, 0.5f, 0.5f, 1});
      if (FLB::Renderer::s_UpdateRender)
      {
	// variable that will be used as a pointer to access the vertex buffer object from CUDA code
	PRECISION* cudaUx;
	// Map the CUDA graphics resource to a CUDA device pointer
	cudaGraphicsMapResources(1, &cudaVertexBufferUx, 0);
	size_t memorySizeUx;
	cudaGraphicsResourceGetMappedPointer((void**)&cudaUx, &memorySizeUx, cudaVertexBufferUx);

	//TEMPORAL
	float r0;
	r0 = ((double)(rand())/RAND_MAX);


	if (!FLB::Renderer::s_VelocityPointMode) 
	{
	  cudaGraphicsMapResources(1, &cudaTexture, 0);
	  cudaArray* texTureCudaArray;
	  cudaGraphicsSubResourceGetMappedArray(&texTureCudaArray, cudaTexture, 0, 0);

	  struct cudaResourceDesc resourceDesc;
	  memset(&resourceDesc, 0, sizeof(resourceDesc));

	  resourceDesc.resType = cudaResourceTypeArray;
	  resourceDesc.res.array.array = texTureCudaArray;

	  cudaCreateSurfaceObject(&d_SurfaceTexture, &resourceDesc);
	  vPrueba2<PRECISION><<<1, 1>>>(r0, cudaUx, d_SurfaceTexture);
	  cudaGraphicsUnmapResources(1, &cudaTexture);
	}
	//TEMPORAL
	if(FLB::Renderer::s_VelocityPointMode) vPrueba<PRECISION><<<numBlocks,threadsPerBlock>>>(r0, cudaUx);
	cudaDeviceSynchronize();
	//checkCudaErrors(cudaGetLastError());


	// Unmap the vertex buffer object so that OpenGL can access the resource
	cudaGraphicsUnmapResources(1, &cudaVertexBufferUx);

	 
      }
      FLB::Renderer::clear();
      //FLB::Renderer::beginScene();
      FLB::Renderer::submit(vertexArray, textureShader, numPointsMesh, orthographicCameraController.getCamera().getViewProjectionMatrix());
      //FLB::Renderer::endScene();
      renderWindow.update();

      //sleep(1);
    }
    //cleanup
    cudaGraphicsUnregisterResource(cudaVertexBufferUx);
    cudaGraphicsUnregisterResource(cudaTexture);
    cudaDestroySurfaceObject(d_SurfaceTexture);
  }
  /*else (optionsCalc.typeProblem == FLB::TypeProblem::MONOFLUID && optionsCalc.graphicsAPI == FLB::RendererAPI::NONE)
  {
    while (t < maxIterations)
    {

    }

  }*/


  //cudaFree(void *devPtr);

}

// Explicit instantation of the functions because the templates functions are instantiated in a diferent compilation unit
template void FLB::h_runCudaCalculations2D<float>(FLB::OptionsCalculation& optionsCalc, FLB::Window& renderWindow, FLB::VertexArray& vertexArray, FLB::Shader& textureShader, FLB::OrthographicCameraController& orthographicCameraController, float* h_weights, FLB::Mesh* mesh, size_t maxIterations, unsigned int numDimensions, unsigned int numVelocities, Fl_Simple_Terminal* terminal);

//template void FLB::h_runCudaCalculations2D<double>(FLB::OptionsCalculation& optionsCalc, FLB::Window& renderWindow, FLB::VertexArray& vertexArray, FLB::Shader& textureShader, FLB::OrthographicCamera& camera, glm::vec3 cameraPosition, double* h_weights, size_t numPointsMesh, size_t maxIterations, unsigned int numDimensions, unsigned int numVelocities);


