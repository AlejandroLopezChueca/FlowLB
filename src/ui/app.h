#pragma once

#include "geometry/mesh.h"
#include "io/reader.h"
#include "graphics/window.h"

#include "FL/Fl_Box.H"
#include "FL/Fl_Input.H"
#include "FL/Fl_Widget.H"
#include <FL/Fl.H>
#include <FL/Fl_Window.H>
#include <FL/Fl_Button.H>
#include <FL/Enumerations.H>
#include <FL/Fl_Simple_Terminal.H>
#include <FL/Fl_Native_File_Chooser.H>
#include <FL/Fl_Output.H>
#include <functional>
#include <memory>


namespace FLB
{
  class App
  {
    public:
      App();
      //~App();

      void startApp();
      
      static void closeGraphics() {s_RunningGraphics = false;}


      //static FLB::GeometryReader* geometryReader;
      //static FLB::CalculationReader* calculationReader;
      static App& getApp() {return *s_Instance;}

      static bool s_AppCreated;
      static bool s_RunningGraphics;


    private:
      static void runCreationDomain(Fl_Widget* widget, void* instance);
      static void runCalculation(Fl_Widget* widget, void* instance);
      static void runPostProcessing(Fl_Widget* widget, void* instance);
      static void getPathFiles(Fl_Widget* widget, void* dirFiles);
      static void clearTerminal(Fl_Widget* widget, void* terminal);
      static void printInfoDevice(Fl_Widget* widget, void* terminal);



      void createDomain();
      void calculate();
      void postProcessing();
 
      std::unique_ptr<Fl_Window> m_MainWindow;
      std::unique_ptr<Fl_Button> m_ButtonCreateDomain;
      std::unique_ptr<Fl_Button> m_ButtonRunCalculation;
      std::unique_ptr<Fl_Button> m_ButtonPostProcessing;
      std::unique_ptr<Fl_Button> m_ButtonDirFiles;
      std::unique_ptr<Fl_Button> m_ButtonClearTerminal;
      std::unique_ptr<Fl_Button> m_ButtonPrintInfoDevice;
      std::unique_ptr<Fl_Output> m_DirFiles;
      std::unique_ptr<Fl_Box> m_LabelDir;
      std::unique_ptr<Fl_Box> m_LabelLog;
      std::unique_ptr<Fl_Simple_Terminal> m_Terminal;
      
      FLB::Mesh* m_Mesh;

      static bool s_DomainCreationEnded;
      static bool s_CalculationEnded;
      static bool s_PostProcessingEnded;


    static App* s_Instance;
      //std::unique_ptr<FLB::Window> m_RenderWindow;
      //std::unique_ptr<FLB::OrthographicCameraController> m_OrthographicCameraController;
      
  };

  class StdCapture
  {

  };
}
