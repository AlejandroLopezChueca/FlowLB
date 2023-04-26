#pragma once

#include <array>
#include <cstddef>
#include <string>

namespace FLB
{
  /** Base class for readers.
   *
   */
  class Reader
  {
    public:
      //Reader();
      //~Reader();

    protected:

      /** Split string in two by delimiter.
       * 
       */
      void splitString(std::string OptionsCalc[], std::string line, char delimiter, int posLine);
      /** Check if the option defined by user exist in the options availables.
       *
       */
      template<std::size_t N>
      bool checkOptionExistence(std::string option, std::array<std::string, N> listOptions);

  };

  /** Reader of the mesh.
   *
   */
  class MeshReader: public Reader
  {
    public:
      //MeshReader();
      //~MeshReader();

      void readMesh(std::string meshPath);

  };

  enum class RendererAPI
  {
    NONE = 0, OPENGL = 1
  };
  /** Options to use in the calculations.
   *
   */
  struct OptionsCalculation
  {
    int typeAnalysis;  /**< 0=2D 1=3D. */
    bool timeAnalysis;
    bool plotGraphics;
    RendererAPI graphicsAPI;
    int precision;
    float flow;
    float maxError;
  };

  /** Reader of the options of the calculations.
   *
   */
  class CalculationReader: public Reader
  {
    public:
      //CalculationReader();
      //~CalculationReader();
      
      /** Read the file with the options of the calculation.
       *
       */
      void readOptionsCalculation(std::string filePath, OptionsCalculation& OptionsCalc);
      /** Get the option from the file with the options.
       *
       */
      void getOptionCalculation(std::string option, std::string value, OptionsCalculation& OptionsCalc, int posLine);

  };
      
}
