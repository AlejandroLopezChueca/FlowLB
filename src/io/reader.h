#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "FL/Fl_Simple_Terminal.H"
#include "geometry/mesh.h"
#include "graphics/API.h"
#include "utils.h"

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
       * @ return Return false if an error has occurred
       */
      bool splitString(std::string OptionsCalc[], std::string line, char delimiter, unsigned int posLine, Fl_Simple_Terminal* terminal, int typeError);

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

  struct CudaBlockSize
  {
    uint32_t x = 0, y = 0, z = 0;
  };


  /** Options to use in the calculations.
   *
   */
  struct OptionsCalculation
  {
    int typeAnalysis = 0;  // 0=2D 1=3D
    CalculationAPI calculationAPI = CalculationAPI::CUDA;
    CudaBlockSize cudaBlockSize;
    API graphicsAPI = API::NONE;
    TypeProblem typeProblem = TypeProblem::MONOFLUID;
    uint8_t collisionOperator = 0; // 0 = SRT, 1 = MRT
    bool surfaceTension = false;
    int precision = 32;
    double LBVelocity = 0.1; // the lattice velocity must be less than cs^2 (sound velocity in lattice units) that it is tipically 1/3 (depend on the lattice scheme)
    bool useGravity = false;
    float gravity = 9.80665;
    double flow = 0.0;
    double velocity = 0.0;
    double kinematicViscosity = 0.0;
    double density = 0.0;
    float timeSimulation = 0.0f;
    float timeSave = 0.0f;

    FLB::TypesNodes boundaryLeft = FLB::TypesNodes::NONE;
    FLB::TypesNodes boundaryRight = FLB::TypesNodes::NONE;
    FLB::TypesNodes boundaryUp = FLB::TypesNodes::NONE;
    FLB::TypesNodes boundaryDown = FLB::TypesNodes::NONE;
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
      bool readOptionsCalculation(std::string& filePath, FLB::OptionsCalculation& optionsCalc, Fl_Simple_Terminal* terminal);

      /**
       * @brief Read some options in the file of the calculation's options
       *
       */
      bool readSomeOptionsCalculation(std::string& filePath, FLB::OptionsCalculation& optionsCalc, Fl_Simple_Terminal* terminal, std::vector<std::string>& optionsToSearch);
     
    private:
      /**
       * @brief Last check before finishing the reading of the data and that it cannot be done before having read the entire file
       *
       */
      bool checkOptionsCalculation(FLB::OptionsCalculation& optionsCalc, Fl_Simple_Terminal* terminal, bool checkAllOptions = true);

      /**
       *  @brief Check if a option in the file exist and extract it's value. It also throws an error if the option doesn't exsit 
       *
       */
      bool getOptionCalculation(std::string& option, std::string& value, FLB::OptionsCalculation& optionsCalc, unsigned int posLine, Fl_Simple_Terminal* terminal);

      /**
       * @brief Modify some of the data when the reading of the file has finished because it is necesary to know all the data.
       *
       */
      void modifyCalculationData(FLB::Mesh* mesh);

      void getTypeNode(FLB::TypesNodes& typeNode, std::string& value);
  };

  /**
   *  @brief Reader for the geometry options
   *
   */
  class GeometryReader: public Reader
  {
    public:
      GeometryReader(std::filesystem::path directoryPath);


      /**
       *  @brief Main function to start to read all the options in the file of the geometry
       *
       *  @param[in]
       *
       */
      bool readGeometryOptions(std::string filePath, FLB::Mesh* mesh, Fl_Simple_Terminal* terminal);

    private:
      /**
       * @brief Last check before finishing the reading of the data and that it cannot be done before having readed the entire file
       *
       */
      bool checkOptionsGeometry(FLB::Mesh* mesh, Fl_Simple_Terminal* terminal);

      /**
       * @brief Get all the elements that make up the mesh
       *
       */
      bool getMeshElements(std::string& element, std::string& coordinates, unsigned int posLine, FLB::Mesh* mesh, Fl_Simple_Terminal* terminal);
      
      /**
       *  @brief Get all the data about the cross drainage works 
       *
       */
      bool getCDWData(const std::string& option, const std::string& value, unsigned int posLine, FLB::Mesh* mesh, Fl_Simple_Terminal* terminal);

      /**
       *  @brief Get all the data about the obstacles 
       *
       */
      bool getObstaclesData(const std::string& option, const std::string& value, unsigned int posLine, FLB::Mesh* mesh, Fl_Simple_Terminal* terminal);
      
      /**
       * @brief Modify some of the data when the reading of the file has finished because it is necesary to know all the data.
       *
       */
      bool modifyGeometryData(FLB::Mesh* mesh, Fl_Simple_Terminal* terminal);
      
      std::filesystem::path m_DirectoryPath;
  };
      
}
