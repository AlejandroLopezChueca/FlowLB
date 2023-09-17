#pragma once

#include <array>
#include <cstddef>
#include <string>

#include "geometry/mesh.h"
#include "graphics/API.h"

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

      /** 
       * @brief Split string in two by delimiter.
       * 
       */
      void splitString(std::string OptionsCalc[], std::string line, char delimiter, unsigned int posLine);

      /** 
       * @brief Check if the option defined by user exist in the options availables.
       *
       */
      template<std::size_t N>
      bool checkOptionExistence(std::string option, std::array<std::string, N> listOptions);

      /**
       * @brief Extract string from start to character or from character to end. If the character doesn't exist, it return the same string
       */
      static std::string extractString(std::string option, char delimiter, unsigned int posLine, bool toEnd = false);
       
      /**
       * @brief Extract string beetween two characters
       */
      static std::string extractString(std::string option, char initDelimiter, char endDelimiter, unsigned int posLine);

      /**
       * @brief Extract string beetween two characters. It accepts to search ocurrences that are not the first
       *
       *  @param[in]  option
       *  @param[in]  initDelimiter
       *  @param[in]  endDelimiter
       *  @param[in]  positionInitChar  Ocurrence to search of the firts delimiter
       *  @param[in]  positionEndChar   Ocurrence yo search of the end delimiter
       *  @param[in]  posLine
       *
       *  @return      
       */
      static std::string extractString(std::string option, char initDelimiter, char endDelimiter, int positionInitChar, int positionEndChar, unsigned int posLine);
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

  enum class TypeProblem
  {
    MONOFLUID = 0, FREE_SURFACE = 1
  };
  /** Options to use in the calculations.
   *
   */
  struct OptionsCalculation
  {
    int typeAnalysis = 0;  // 0=2D 1=3D
    API graphicsAPI = API::NONE;
    TypeProblem typeProblem = TypeProblem::MONOFLUID;
    bool surfaceTension = false;
    int precision = 32;
    float flow = 0.0f;
    float kinematicViscosity = 0.0f;
    float timeSimulation = 0.0f;
    float timeSave = 0.0f;
  };

  /** 
   *  @brief Reader of the options of the calculations.
   */
  class CalculationReader: public Reader
  {
    public:
      //CalculationReader();
      //~CalculationReader();
      
      /**
       *  @brief Main function to start to read all the options in the file of the calculation's options
       *
       */
      void readOptionsCalculation(std::string filePath, FLB::OptionsCalculation& optionsCalc);
      
      /**
       * @brief Last check before finishing the reading of the data and that it cannot be done before having readed the entire file
       *
       */
      void checkOptionsCalculation(FLB::OptionsCalculation& optionsCalc);

      /**
       *  @brief Check if a option in the file exist and extract it's value. It also throws an error if the option doesn't exsit 
       *
       */
      void getOptionCalculation(std::string option, std::string value, FLB::OptionsCalculation& OptionsCalc, unsigned int posLine);

      /**
       * @brief Modify some of the data when the reading of the file has finished because it is necesary to know all the data.
       *
       */
      void modifyCalculationData(FLB::Mesh* mesh);
  };

  /**
   *  @brief Reader for the geometry options
   *
   */
  class GeometryReader: public Reader
  {
    public:


      /**
       *  @brief Main function to start to read all the options in the file of the geometry
       *
       *  @param[in]
       *
       */
      void readGeometryOptions(std::string filePath, FLB::Mesh* mesh);

    private:
      /**
       * @brief Last check before finishing the reading of the data and that it cannot be done before having readed the entire file
       *
       */
      void checkOptionsGeometry(FLB::Mesh* mesh);

      /**
       * @brief Get all the elements that make up the mesh
       *
       */
      void getMeshElements(std::string element, std::string coordinates, unsigned int posLine, FLB::Mesh* mesh);
      
      /**
       *  @brief Get all the data about the cross drainage works 
       *
       */
      void getCDWData(std::string option, std::string value, unsigned int posLine, FLB::Mesh* mesh);

      /**
       *  @brief Get all the data about the obstacles 
       *
       */
      void getObstaclesData(std::string option, std::string value, unsigned int posLine, FLB::Mesh* mesh);
      
      /**
       * @brief Modify some of the data when the reading of the file has finished because it is necesary to know all the data.
       *
       */
      void modifyGeometryData(FLB::Mesh* mesh);
  };
      
}
