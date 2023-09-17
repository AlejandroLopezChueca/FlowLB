#include "FL/Enumerations.H"
#include "FL/Fl_Box.H"
#include "FL/Fl_Button.H"
#include "FL/Fl_Input.H"
#include "FL/Fl_Native_File_Chooser.H"
#include "FL/Fl_Output.H"
#include "FL/Fl_Simple_Terminal.H"
#include "FL/Fl_Widget.H"
#include "FL/Fl_Window.H"
#include <iostream>
#include <memory>
#include <string>
#include <cuda_runtime.h>
#include <filesystem>

#include "app.h"
#include "geometry/mesh.h"
#include "graphics/renderer.h"
#include "graphics/shader.h"
#include "graphics/buffer.h"
#include "graphics/vertexArray.h"
#include "graphics/renderer.h"
#include "graphics/rendererAPI.h"
#include "graphics/texture.h"
#include "graphics/frameBuffer.h"
#include "cuda/runCuda.cuh"
#include "initData.h"
#include "io/writer.h"

FLB::App::App()
{
  float space = 10.0f;
  float widthButton = (width - 4*space)/3;
  float heightButton = 40.0f;
  float acumHeight = 0.0f;
  float heightDirFiles = 30.0f;
  float widthDirFiles = widthButton * 3 *5/6;
  float heightLabelLog = 40.0f;
 
  //mesh = new FLB::Mesh();
  mainWindow = new Fl_Window(width, height, "FlowLB");
  buttonCreateDomain = new Fl_Button(space, 10, widthButton, heightButton, "Create Domain");
  buttonRunCalculation = new Fl_Button(2*space + widthButton, 10, widthButton, heightButton, "Run");
  buttonPostProcessing = new Fl_Button(3*space + 2*widthButton, 10, widthButton, heightButton, "PostProcessing");
  acumHeight += space + heightButton + 2*space;
  labelDir = new Fl_Box(space, acumHeight, widthButton, heightButton/2, "Directory");
  acumHeight += heightButton/2;
  dirFiles = new Fl_Output(space, acumHeight, widthDirFiles, heightDirFiles);
  buttonDirFiles = new Fl_Button(2*space + widthDirFiles, acumHeight, 3*widthButton*1/6, heightDirFiles, "...");
  //buttonDirFiles = new WidgetWrapper<Fl_Button>(2*space + widthDirFiles, acumHeight, 3*widthButton*1/6, heightDirFiles, "...");
  acumHeight += heightDirFiles + space;
  labelLog = new Fl_Box(space, acumHeight, width -2*space, heightLabelLog, "LOG");
  buttonClearTerminal = new Fl_Button(2*space + widthDirFiles, acumHeight + 0.5*(heightLabelLog - heightDirFiles), 3*widthButton*1/6, heightDirFiles, "Clear Log");
  acumHeight += heightLabelLog;
  terminal = new Fl_Simple_Terminal(space, acumHeight, width -2*space, height - acumHeight - space); 
}

FLB::App::~App()
{
  //delete mesh;
  delete buttonCreateDomain;
  delete buttonRunCalculation;
  delete buttonPostProcessing;
  delete labelDir;
  delete dirFiles;
  delete buttonDirFiles;
  delete labelLog;
  delete terminal;
  delete mainWindow;
}

bool FLB::App::calculationEnded = true;
bool FLB::App::domainCreationEnded = true;
bool FLB::App::appCreated = false;

void FLB::App::startApp()
{
  // We tell FLTK that we will not add any more widgets the the main window
  mainWindow -> end();
  // We show the window
  mainWindow -> show();

  //Theming
  Fl::background(255, 255, 255);
  Fl::visible_focus(false);

  mainWindow -> labelfont(FL_HELVETICA_BOLD);
  buttonCreateDomain -> labelfont(FL_HELVETICA_BOLD);
  buttonCreateDomain -> color(BLUE);
  buttonCreateDomain -> selection_color(SEL_BLUE);
  buttonCreateDomain -> labelcolor(FL_WHITE);
  buttonCreateDomain -> box(FL_GTK_ROUND_UP_BOX);

  buttonRunCalculation -> labelfont(FL_HELVETICA_BOLD);
  buttonRunCalculation -> color(BLUE);
  buttonRunCalculation -> selection_color(SEL_BLUE);
  buttonRunCalculation -> labelcolor(FL_WHITE);
  buttonRunCalculation -> box(FL_GTK_ROUND_UP_BOX);

  buttonPostProcessing -> labelfont(FL_HELVETICA_BOLD);
  buttonPostProcessing -> color(BLUE);
  buttonPostProcessing -> selection_color(SEL_BLUE);
  buttonPostProcessing -> labelcolor(FL_WHITE);
  buttonPostProcessing -> box(FL_GTK_ROUND_UP_BOX);

  labelDir -> labelfont(FL_HELVETICA_BOLD);
  labelDir -> labelsize(18);
  labelDir -> color(FL_WHITE);
  labelDir -> labelcolor(BLUE);
  labelDir -> align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE);

  dirFiles -> box(FL_DOWN_BOX);

  buttonDirFiles -> labelfont(FL_HELVETICA_BOLD);
  buttonDirFiles -> box(FL_GTK_UP_BOX);

  buttonClearTerminal -> labelfont(FL_HELVETICA_BOLD);
  buttonClearTerminal -> box(FL_GTK_UP_BOX);

  labelLog -> labelfont(FL_HELVETICA_BOLD);
  labelLog -> labelsize(20);
  labelLog -> color(BLUE);
  labelLog -> labelcolor(FL_WHITE);
  labelLog -> box(FL_FLAT_BOX);
  labelLog -> align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE);

  terminal -> ansi(true);
  terminal -> append("\033[47m");
  //terminal -> printf("LOG\n\n");
  
  // Connect buttons with callbacks
  buttonDirFiles-> callback(static_cast<Fl_Callback*>(FLB::App::getPathFiles), dirFiles);
  buttonCreateDomain-> callback(static_cast<Fl_Callback*>(FLB::App::runCreationDomain), this);
  buttonClearTerminal -> callback(static_cast<Fl_Callback*>(FLB::App::clearTerminal), terminal);
  buttonRunCalculation-> callback(static_cast<Fl_Callback*>(FLB::App::runCalculation), this);


  Fl::run();
}

void FLB::App::clearTerminal(Fl_Widget *widget, void *termi)
{
  Fl_Simple_Terminal* thisTerminal = static_cast<Fl_Simple_Terminal*>(termi);
  thisTerminal -> clear();

};

void FLB::App::getPathFiles(Fl_Widget* widget, void* dirFiles)
{
  Fl_Output* pathFiles = static_cast<Fl_Output*>(dirFiles);
  Fl_Native_File_Chooser dirChooser;
  dirChooser.title("Select directory with the geometry and options");
  dirChooser.type(Fl_Native_File_Chooser::BROWSE_DIRECTORY);
  switch (dirChooser.show()) 
  {
    case -1: break;
    case 1:  break;
    default:
      pathFiles -> value(dirChooser.filename());
      break;	
  }
};

void FLB::App::runCreationDomain(Fl_Widget* widget, void* instance)
{ 
  if (!domainCreationEnded) return;
  domainCreationEnded = false;
  App* instanceApp = static_cast<App*>(instance);
  // Instantiation of the mesh here to avoid errors when the button is clicked various times when this process is  done in the Constructor
  instanceApp -> mesh = new FLB::Mesh();
  instanceApp -> createDomain();
  domainCreationEnded = true;
  delete instanceApp -> mesh;
};

void FLB::App::runCalculation(Fl_Widget* widget, void* instance)
{
  if (!calculationEnded) return;
  calculationEnded = false;
  App* instanceApp = static_cast<App*>(instance);
  instanceApp -> mesh = new FLB::Mesh();
  instanceApp -> calculate();
  calculationEnded = true;
  delete instanceApp -> mesh; 
};

void FLB::App::runPostProcessing(Fl_Widget *widget, void *instance)
{

}

void FLB::App::createDomain()
{
  std::string geometryPath = dirFiles -> value();
  geometryPath += "/Geometry.txt";
  if (!std::filesystem::exists(geometryPath))
  {
    terminal -> printf("The file Geometry.txt doesn't exist in %s\n", dirFiles -> value());
    return;
  }
  FLB::GeometryReader geometryReader;
  geometryReader.readGeometryOptions(geometryPath, mesh);
  // get number poinst mesh in each direction
  mesh -> getNumberPointsMesh();
  // Reserve memory for arrays of the mesh
  mesh -> reserveMemory();
  //Get the coordinates of each point of the mesh in SI units 
  mesh -> getCoordinatesPoints();
  FLB::initData<float, FLB::voidInitVelocityAndf<float>>(9, mesh);
  // save mesh to disk
  FLB::VTUWriter writer;
  std::string pathSave = dirFiles -> value();
  pathSave += "/mesh.vtu";
  writer.addField<uint8_t>("UInt8", "Flags", mesh -> domainFlags, mesh -> numPointsMesh, true);
  writer.savePointData(pathSave, mesh, terminal); 
}

void FLB::App::calculate()
{
  std::string geometryPath = dirFiles -> value();
  std::string optionsCalculationPath = dirFiles -> value();
  geometryPath += "/Geometry.txt";
  optionsCalculationPath += "/OptionsCalc.txt";
  if (!std::filesystem::exists(geometryPath))
  {
    terminal -> printf("The file Geometry.txt doesn't exist in %s\n", dirFiles -> value());
    return;
  }
  if (!std::filesystem::exists(optionsCalculationPath))
  {
    terminal -> printf("The file OptionsCalc.txt doesn't exist in %s\n", dirFiles -> value());
    return;
  }

  //Reading the geometry's options, but before it is neccesary to clear the data members vectors
  mesh -> clear();
  FLB::GeometryReader geometryReader;
  geometryReader.readGeometryOptions(geometryPath, mesh);
  // get number poinst mesh in each direction
  mesh -> getNumberPointsMesh();
  // Reserve memory for arrays of the mesh
  mesh -> reserveMemory();
  //Get the coordinates of each point of the mesh in SI units 
  mesh -> getCoordinatesPoints();
  // Indices of the mesh to create the domain of the graphics
  mesh -> getIndicesCorners();

  //Reading the calculation's Options
  FLB::OptionsCalculation optionsCalc; 
  FLB::CalculationReader calcReader;
  //calcReader.readOptionsCalculation(optionsCalculationPath, optionsCalc);

  // Graphics options
  optionsCalc.graphicsAPI = FLB::API::OPENGL;

  // Set the API to use for the render
  FLB::RendererAPI::setAPI(optionsCalc.graphicsAPI);
  // Set the renderer to call the APi
  FLB::Renderer::setRendererAPI(mesh -> m_indicesCorners);
  // seto options used in 2D or 3D
  std::unique_ptr<FLB::Window> m_RenderWindow;
  std::unique_ptr<FLB::OrthographicCameraController> m_OrthographicCameraController;
  if (optionsCalc.typeAnalysis == 0)
  {
    m_OrthographicCameraController.reset(new FLB::OrthographicCameraController(1600.0f, 900.0f));
    m_RenderWindow.reset(FLB::Window::create<FLB::OrthographicCameraController>(optionsCalc.graphicsAPI, m_OrthographicCameraController.get(), optionsCalc.typeAnalysis, terminal));
  }

  // First it is neccesary to create and bind the vertex array
  std::unique_ptr<FLB::VertexArray> vertexArray = FLB::VertexArray::create(optionsCalc.graphicsAPI);
  std::unique_ptr<FLB::VertexBuffer> vertexBufferCoordinates = FLB::VertexBuffer::create(optionsCalc.graphicsAPI, terminal, mesh -> coordinatesPoints.data(), 3 * mesh -> numPointsMesh * sizeof(float));
  FLB::BufferLayout layout = {
    {ShaderDataType::Float3, "a_PointsPosition"}
  };
  vertexBufferCoordinates -> setLayout(layout);
  vertexArray -> addVertexBuffer(vertexBufferCoordinates.get());
  FLB::Renderer::createQuad(optionsCalc.graphicsAPI, terminal);
  vertexArray -> addVertexBuffer(FLB::Renderer::s_VertexBufferQuad.get());

  std::unique_ptr<FLB::Texture1D> texture1D = FLB::Texture1D::create(optionsCalc.graphicsAPI);
  texture1D->bind(0);

  std::unique_ptr<FLB::Shader> shaderPoinstVelocity = FLB::Shader::create("assets/shaders/pointsVelocity.glsl", optionsCalc.graphicsAPI, terminal);
  shaderPoinstVelocity -> setInt("u_ColorMap", 0);// slot 0
  std::unique_ptr<FLB::Shader> shaderTextureVelocity2D = FLB::Shader::create("assets/shaders/textureVelocity2D.glsl", optionsCalc.graphicsAPI, terminal);
  //shaderTextureVelocity2D -> setInt("u_ColorMap", 0);// slot 0
  shaderTextureVelocity2D -> setInt("u_Texture", 0);// slot 0
 
  FLB::Renderer::s_shaderToUse = shaderPoinstVelocity.get(); //dafault
  FLB::Renderer::s_shaderPointsVelocity = shaderPoinstVelocity.get();
  FLB::Renderer::s_shaderTextureVelocity2D = shaderTextureVelocity2D.get();

  // Textures
  FLB::FrameBufferSpecifications specs;
  specs.width = mesh -> Nx;
  specs.height = mesh -> Ny;

  std::unique_ptr<FLB::FrameBuffer> a = FLB::FrameBuffer::create(optionsCalc.graphicsAPI, specs);
  FLB::Renderer::s_TextureToUse = a.get();
  FLB::Renderer::s_TextureVelocity = a.get();


  unsigned int numDimensions;
  unsigned int numVelocities;

  size_t maxIterations = optionsCalc.timeSimulation;
  
  if (optionsCalc.precision == 32)
  {
    float h_weights[9] = {4.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f};
    if (optionsCalc.typeAnalysis == 0) //2D
    {
      numDimensions = 2;
      numVelocities = 9;
      FLB::h_runCudaCalculations2D<float>(optionsCalc, *m_RenderWindow, *vertexArray, *shaderPoinstVelocity, *m_OrthographicCameraController, h_weights, mesh, maxIterations, numDimensions, numVelocities, terminal);
    }
    else //3D Analysis
    {
    
    }
  }
  
  else if (optionsCalc.precision == 64)
  {

    double h_weights[9] = {4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};

  if (optionsCalc.precision == 32)
  {

  }

  
    //FLB::h_runCudaCalculations2D<double>(optionsCalc, h_weights, mesh -> numPointsMesh, maxIterations, numDimensions, numVelocities);
  }
  // generate Error in window
  //m_OrthographicCameraController.reset();
  //m_RenderWindow.reset(m_RenderWindow.get());
}

void FLB::App::postProcessing()
{

}
