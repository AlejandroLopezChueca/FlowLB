#pragma once

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

#include "geometry/mesh.h"
//#include "io/reader.h"
#include "graphics/window.h"

#define BLUE fl_rgb_color(0x42, 0xA5, 0xF5)
#define SEL_BLUE fl_rgb_color(0x21, 0x96, 0xF3)

namespace FLB
{
  class App
  {
    public:
      App();
      ~App();

      void startApp();
      static void runCreationDomain(Fl_Widget* widget, void* instance);
      static void runCalculation(Fl_Widget* widget, void* instance);
      static void runPostProcessing(Fl_Widget* widget, void* instance);
      static void getPathFiles(Fl_Widget* widget, void* dirFiles);
      static void clearTerminal(Fl_Widget* widget, void* termi);

      void createDomain();
      void calculate();
      void postProcessing();

      Fl_Window* mainWindow;
      Fl_Button* buttonCreateDomain;
      Fl_Button* buttonRunCalculation;
      Fl_Button* buttonPostProcessing;
      Fl_Button* buttonDirFiles;
      Fl_Button* buttonClearTerminal;
      Fl_Output* dirFiles;
      Fl_Box* labelDir;
      Fl_Box* labelLog;
      Fl_Simple_Terminal* terminal;
      
      FLB::Mesh* mesh;
      //static FLB::GeometryReader* geometryReader;
      //static FLB::CalculationReader* calculationReader;

      static const int width = 780;
      static const int height = 500;

      static bool domainCreationEnded;
      static bool calculationEnded;
      bool postProcessingEnded = true;

      static bool appCreated;

      //std::string pathFiles;
    private:
      //std::unique_ptr<FLB::Window> m_RenderWindow;
      //std::unique_ptr<FLB::OrthographicCameraController> m_OrthographicCameraController;
  };

  class StdCapture
  {

  };
}
