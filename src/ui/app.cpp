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
#include <vector>


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
#include "renderLayer.h"
#include "cuda/runCuda.cuh"
#include "cuda/cudaUtils.cuh"
#include "graphics/window.h"
#include "initData.h"
#include "io/reader.h"
#include "io/writer.h"
#include "renderLayer.h"


#define BLUE fl_rgb_color(0x42, 0xA5, 0xF5)
#define SEL_BLUE fl_rgb_color(0x21, 0x96, 0xF3)

FLB::App* FLB::App::s_Instance = nullptr;

FLB::App::App()
{
  s_Instance = this;
  const int width = 780;
  const int height = 500;

  const float space = 10.0f;
  const float widthButton = (width - 4*space)/3;
  const float heightButton = 40.0f;
  const float heightDirFiles = 30.0f;
  const float widthDirFiles = widthButton * 3 *5/6;
  const float heightLabelLog = 40.0f;
  
  float acumHeight = 0.0f;
 
  //mesh = new FLB::Mesh();
  m_MainWindow = std::make_unique<Fl_Window>(width, height, "FlowLB");
  m_ButtonCreateDomain = std::make_unique<Fl_Button>(space, 10, widthButton, heightButton, "Create Domain");
  m_ButtonRunCalculation = std::make_unique<Fl_Button>(2*space + widthButton, 10, widthButton, heightButton, "Run");
  m_ButtonPostProcessing = std::make_unique<Fl_Button>(3*space + 2*widthButton, 10, widthButton, heightButton, "PostProcessing");
  acumHeight += space + heightButton + 2*space;
  m_LabelDir = std::make_unique<Fl_Box>(space, acumHeight, widthButton, heightButton/2, "Directory");
  acumHeight += heightButton/2;
  m_DirFiles = std::make_unique<Fl_Output>(space, acumHeight, widthDirFiles, heightDirFiles);
  m_ButtonDirFiles = std::make_unique<Fl_Button>(2*space + widthDirFiles, acumHeight, 3*widthButton*1/6, heightDirFiles, "...");
  //buttonDirFiles = new WidgetWrapper<Fl_Button>(2*space + widthDirFiles, acumHeight, 3*widthButton*1/6, heightDirFiles, "...");
  acumHeight += heightDirFiles + space;
  m_LabelLog = std::make_unique<Fl_Box>(space, acumHeight, width -2*space, heightLabelLog, "LOG");
  m_ButtonClearTerminal = std::make_unique<Fl_Button>(2*space + widthDirFiles, acumHeight + 0.5*(heightLabelLog - heightDirFiles), 3*widthButton*1/6, heightDirFiles, "Clear Log");
  m_ButtonPrintInfoDevice = std::make_unique<Fl_Button>(2*space + 0.5 * widthDirFiles, acumHeight + 0.5*(heightLabelLog - heightDirFiles), 3*widthButton*1/6, heightDirFiles, "Print GPU Info");
  acumHeight += heightLabelLog;
  m_Terminal = std::make_unique<Fl_Simple_Terminal>(space, acumHeight, width -2 * space, height - acumHeight - space); 
}

bool FLB::App::s_CalculationEnded = true;
bool FLB::App::s_DomainCreationEnded = true;
bool FLB::App::s_AppCreated = false;
bool FLB::App::s_PostProcessingEnded = false;
bool FLB::App::s_RunningGraphics = true;

void FLB::App::startApp()
{
  // We tell FLTK that we will not add any more widgets than the main window
  m_MainWindow -> end();
  // We show the window
  m_MainWindow -> show();

  //Theming
  Fl::background(255, 255, 255);
  Fl::visible_focus(false);

  m_MainWindow -> labelfont(FL_HELVETICA_BOLD);
  m_ButtonCreateDomain -> labelfont(FL_HELVETICA_BOLD);
  m_ButtonCreateDomain -> color(BLUE);
  m_ButtonCreateDomain -> selection_color(SEL_BLUE);
  m_ButtonCreateDomain -> labelcolor(FL_WHITE);
  m_ButtonCreateDomain -> box(FL_GTK_ROUND_UP_BOX);

  m_ButtonRunCalculation -> labelfont(FL_HELVETICA_BOLD);
  m_ButtonRunCalculation -> color(BLUE);
  m_ButtonRunCalculation -> selection_color(SEL_BLUE);
  m_ButtonRunCalculation -> labelcolor(FL_WHITE);
  m_ButtonRunCalculation -> box(FL_GTK_ROUND_UP_BOX);

  m_ButtonPostProcessing -> labelfont(FL_HELVETICA_BOLD);
  m_ButtonPostProcessing -> color(BLUE);
  m_ButtonPostProcessing -> selection_color(SEL_BLUE);
  m_ButtonPostProcessing -> labelcolor(FL_WHITE);
  m_ButtonPostProcessing -> box(FL_GTK_ROUND_UP_BOX);

  m_LabelDir -> labelfont(FL_HELVETICA_BOLD);
  m_LabelDir -> labelsize(18);
  m_LabelDir -> color(FL_WHITE);
  m_LabelDir -> labelcolor(BLUE);
  m_LabelDir -> align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE);

  m_DirFiles -> box(FL_DOWN_BOX);

  m_ButtonDirFiles -> labelfont(FL_HELVETICA_BOLD);
  m_ButtonDirFiles -> box(FL_GTK_UP_BOX);

  m_ButtonClearTerminal -> labelfont(FL_HELVETICA_BOLD);
  m_ButtonClearTerminal -> box(FL_GTK_UP_BOX);
  
  m_ButtonPrintInfoDevice -> labelfont(FL_HELVETICA_BOLD);
  m_ButtonPrintInfoDevice -> box(FL_GTK_UP_BOX);

  m_LabelLog -> labelfont(FL_HELVETICA_BOLD);
  m_LabelLog -> labelsize(20);
  m_LabelLog -> color(BLUE);
  m_LabelLog -> labelcolor(FL_WHITE);
  m_LabelLog -> box(FL_FLAT_BOX);
  m_LabelLog -> align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE);

  m_Terminal -> ansi(true);
  m_Terminal -> append("\033[47m");
  //terminal -> printf("LOG\n\n");
  
  // Connect buttons with callbacks
  m_ButtonDirFiles -> callback(static_cast<Fl_Callback*>(FLB::App::getPathFiles), m_DirFiles.get());
  m_ButtonCreateDomain -> callback(static_cast<Fl_Callback*>(FLB::App::runCreationDomain), this);
  m_ButtonClearTerminal -> callback(static_cast<Fl_Callback*>(FLB::App::clearTerminal), m_Terminal.get());
  m_ButtonPrintInfoDevice -> callback(static_cast<Fl_Callback*>(FLB::App::printInfoDevice), m_Terminal.get());
  m_ButtonRunCalculation -> callback(static_cast<Fl_Callback*>(FLB::App::runCalculation), this);


  Fl::run();
}

void FLB::App::clearTerminal(Fl_Widget *widget, void *terminal)
{
  Fl_Simple_Terminal* thisTerminal = static_cast<Fl_Simple_Terminal*>(terminal);
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
  if (!s_DomainCreationEnded) return;
  s_DomainCreationEnded = false;
  App* instanceApp = static_cast<App*>(instance);
  // Instantiation of the mesh here to avoid errors when the button is clicked various times when this process is  done in the Constructor
  instanceApp -> m_Mesh = new FLB::Mesh();
  instanceApp -> createDomain();
  s_DomainCreationEnded = true;
  delete instanceApp -> m_Mesh;
};

void FLB::App::runCalculation(Fl_Widget* widget, void* instance)
{
  if (!s_CalculationEnded) return;
  s_CalculationEnded = false;
  s_RunningGraphics = true; // reset value
  App* instanceApp = static_cast<App*>(instance);
  instanceApp -> m_Mesh = new FLB::Mesh();
  instanceApp -> calculate();
  s_CalculationEnded = true;
  delete instanceApp -> m_Mesh; 
};

void FLB::App::runPostProcessing(Fl_Widget *widget, void *instance)
{

}

void FLB::App::createDomain()
{
  std::string geometryPath = m_DirFiles -> value();
  std::string optionsCalculationPath = m_DirFiles -> value();
  geometryPath += "/Geometry.txt";
  optionsCalculationPath += "/OptionsCalc.txt";
  if (!std::filesystem::exists(geometryPath))
  {
    m_Terminal -> printf("The file Geometry.txt doesn't exist in %s\n", m_DirFiles -> value());
    return;
  }
  if (!std::filesystem::exists(optionsCalculationPath))
  {
    m_Terminal -> printf("[ERROR] The file OptionsCalc.txt doesn't exist in %s\n", m_DirFiles -> value());
    return;
  }

  // Read geometry options
  m_Mesh -> clear();
  FLB::GeometryReader geometryReader(m_DirFiles -> value());
  if (!geometryReader.readGeometryOptions(geometryPath, m_Mesh, m_Terminal.get())) return;
  m_Mesh -> init();

  // Read some calculation options
  FLB::OptionsCalculation optionsCalc; 
  FLB::CalculationReader calcReader(m_DirFiles -> value());
  std::vector<std::string> optionsToSearch = {"TYPE_PROBLEM", "LEFT_BOUNDARY", "RIGHT_BOUNDARY", "UP_BOUNDARY", "DOWN_BOUNDARY"};
  if (!calcReader.readSomeOptionsCalculation(optionsCalculationPath, optionsCalc, m_Terminal.get(), m_Mesh, optionsToSearch)) return;
 
  std::vector<float> voidVector; // It is not used
  FLB::initData<float, FLB::voidInitFields<float>, FLB::voidInitFreeSurfaceFields<float>>(9, m_Mesh, voidVector, voidVector, voidVector, voidVector, voidVector, voidVector, voidVector, &optionsCalc);
  // save mesh to disk
  std::filesystem::path directorySave = std::filesystem::path(m_DirFiles -> value());
  FLB::VTIWriter writer{directorySave, true};
  writer.addField<uint8_t>("UInt8", "Flags", m_Mesh -> getDomainFlags(), m_Mesh -> getNumberPointsMesh(), true);
  writer.writeData(m_Mesh, m_Terminal.get(), true);
}

void FLB::App::calculate()
{
  std::string geometryPath = m_DirFiles -> value();
  std::string optionsCalculationPath = m_DirFiles -> value();
  geometryPath += "/Geometry.txt";
  optionsCalculationPath += "/OptionsCalc.txt";
  if (!std::filesystem::exists(geometryPath))
  {
    m_Terminal -> printf("[ERROR] The file Geometry.txt doesn't exist in %s\n", m_DirFiles -> value());
    return;
  }
  if (!std::filesystem::exists(optionsCalculationPath))
  {
    m_Terminal -> printf("[ERROR] The file OptionsCalc.txt doesn't exist in %s\n", m_DirFiles -> value());
    return;
  }

  //Reading the geometry's options, but before it is neccesary to clear the data members vectors
  m_Mesh -> clear();
  FLB::GeometryReader geometryReader(m_DirFiles -> value());
  if (!geometryReader.readGeometryOptions(geometryPath, m_Mesh, m_Terminal.get())) return;
  m_Mesh -> init();

  //Reading the calculation's Options
  FLB::OptionsCalculation optionsCalc; 
  FLB::CalculationReader calcReader(m_DirFiles -> value());
  if (!calcReader.readOptionsCalculation(optionsCalculationPath, optionsCalc, m_Terminal.get(), m_Mesh)) return;

  unsigned int numDimensions;
  unsigned int numVelocities;

  size_t maxIterations = optionsCalc.timeSimulation;

  // Init render layer here to prevent including entt library in cuda files (create compilation error)
  std::unique_ptr<FLB::RenderLayer> renderLayer;
  FLB::RenderLayer* prtRenderlayer;
  if (optionsCalc.graphicsAPI == FLB::API::OPENGL)
  {
    renderLayer = std::make_unique<FLB::RenderLayer>(m_Mesh, optionsCalc, m_Terminal.get());
    prtRenderlayer = renderLayer.get();
  }
  else prtRenderlayer = nullptr;
  
  std::filesystem::path directorySave = std::filesystem::path(m_DirFiles -> value());
 
  if (optionsCalc.precision == 32)
  {
    std::vector<float> h_weights = {4.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f};
    if (optionsCalc.typeAnalysis == 0) //2D
    {
      numDimensions = 2;
      numVelocities = 9;
      FLB::h_launchCudaCalculations2D<float>(optionsCalc, h_weights, m_Mesh, maxIterations, numDimensions, numVelocities, m_Terminal.get(), prtRenderlayer, directorySave);
    }
    else //3D Analysis
    {
    
    }
  }
  
  else if (optionsCalc.precision == 64)
  {

    double h_weights[9] = {4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};

  if (optionsCalc.typeAnalysis == 0) // 2D
  {

  }

  
    //FLB::h_runCudaCalculations2D<double>(optionsCalc, h_weights, mesh -> numPointsMesh, maxIterations, numDimensions, numVelocities);
  }
  // generate Error in window
  //m_OrthographicCameraController.reset();
  //m_RenderWindow.reset(m_RenderWindow.get());
}


void FLB::App::printInfoDevice(Fl_Widget* widget, void* terminal)
{
  Fl_Simple_Terminal* thisTerminal = static_cast<Fl_Simple_Terminal*>(terminal);
  FLB::CudaUtils::printInfoDevice(thisTerminal);
}

void FLB::App::postProcessing()
{

}



