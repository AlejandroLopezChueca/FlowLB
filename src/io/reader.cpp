
#include "FL/Fl_Simple_Terminal.H"
#include "geometry/mesh.h"
#include "geometry/shapes.h"
#include "graphics/API.h"
#include "math/math.h"
#include "reader.h"
#include "cuda/cudaUtils.cuh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <ostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <algorithm>

////////////////////////////////////// Reader //////////////////////////////////////

bool FLB::Reader::splitString(std::string optionCalc[], std::string textFile, char delimiter, unsigned int posLine, Fl_Simple_Terminal* terminal, int typeError)
{
  std::string typeErrorString;
  if (typeError == 0) typeErrorString = "[GEOMETRY OPTIONS ERROR]";
  else if (typeError == 1) typeErrorString = "[CALCULATION OPTIONS ERROR]";

  // Delete posible comments
  size_t pos = textFile.find_first_of("#", 0);
  if (pos != std::string::npos)
  {
    textFile = textFile.substr(0, pos);
  }

  // The option only has two strings separated by an =, the type and its value
  std::string text;
  std::stringstream lineStream(textFile);
  // Delete whitespaces
  std::regex r("\\s+");

  int i = 0;
  while (std::getline(lineStream, text, delimiter)) 
  {
    if (i > 1) 
    {
      terminal -> printf("%s Error in the line %d\n", typeErrorString.c_str(), posLine);
      return false;
    }
    
    text = std::regex_replace(text, r, "");
    optionCalc[i] = text;  
    i += 1;
  }
  return true;
}

template<std::size_t N>
bool FLB::Reader::checkOptionExistence(std::string option, std::array<std::string, N> listOptions)
{
  return std::find(listOptions.begin(), listOptions.end() , option) != listOptions.end();
}

std::string FLB::Reader::extractString(std::string option, char delimiter, unsigned int posLine, bool toEnd)
{
  std::string value;
  int pos = option.find(delimiter);

  try 
  {
    if (toEnd) value = option.substr(pos + 2, -1); // +2 to cotun the point in the option string
    else value = option.substr(0, pos);
  }
  catch(...)
  {
    throw std::out_of_range("The option in the geometry's file in the line " + std::to_string(posLine) + " is incorrect");
  }
  return value;
}

std::string FLB::Reader::extractString(std::string option, char initDelimiter, char endDelimiter, unsigned int posLine)
{
  std::string value;
  int initPos = option.find(initDelimiter);
  int endPos = option.find(endDelimiter);

  try {value = option.substr(initPos + 1, endPos - initPos - 1);}
  catch(...)
  {
    throw std::out_of_range("The option in the geometry's file in the line " + std::to_string(posLine) + " is incorrect");
  }
  return value;
}

std::string FLB::Reader::extractString(std::string option, char initDelimiter, char endDelimiter, int positionInitChar, int positionEndChar, unsigned int posLine)
{
  int initPos = 0, endPos = 0, count = 0, pos = 0;
  while (count != positionInitChar)
  {
    initPos = option.find(initDelimiter, initPos + pos);
    if (initPos == std::string::npos) throw std::out_of_range("The option in the geometry's file in the line " + std::to_string(posLine) + " is incorrect");
    pos += 1;
    count += 1;
  }

  count = 0;
  pos = 0;
  while (count != positionEndChar)
  {
    endPos = option.find(initDelimiter, endPos + pos);
    if (endPos == std::string::npos) throw std::out_of_range("The option in the geometry's file in the line " + std::to_string(posLine) + " is incorrect");
    pos += 1;
    count += 1;
  }

  std::string value;
  try {value = option.substr(initPos + 1, endPos - initPos - 1);}
  catch(...)
  {
    throw std::out_of_range("[ERROR] The option in the geometry's file in the line " + std::to_string(posLine) + " is incorrect");
  }
  return value;

}

////////////////////////////////////// CalculationReader //////////////////////////////////////

bool FLB::CalculationReader::checkOptionsCalculation(FLB::OptionsCalculation& optionsCalc, Fl_Simple_Terminal* terminal, bool checkAllOptions)
{ 
  if (checkAllOptions && optionsCalc.flow > 1e-15 && optionsCalc.velocity > 1e-15)
  {
    terminal -> printf("[CALCULATION OPTIONS ERROR] Only one of the two arguments, flow and velocity, must be idicated, not both\n");
     return false;
  }
  
  if (checkAllOptions && optionsCalc.flow < 1e-15 && optionsCalc.velocity < 1e-15)
  {
    terminal -> printf("[CALCULATION OPTIONS ERROR] One of the two arguments, flow or velocity, must be idicated\n");
     return false;
  }
  
  if (checkAllOptions && optionsCalc.kinematicViscosity < 1e-12)
  {
    terminal -> printf("[CALCULATION OPTIONS ERROR] The kinematic viscosity has to have a value or be positive\n");
     return false;
  }
  
  if (checkAllOptions && optionsCalc.density < 1e-12)
  {
    terminal -> printf("[CALCULATION OPTIONS ERROR] The density has to have a value or be positive\n");
     return false;
  }

  // check Boundary condtions
  if (optionsCalc.boundaryLeft == FLB::TypesNodes::NONE) 
  {
    terminal -> printf("[CALCULATION OPTIONS ERROR] The boundary condition LEFT_BOUNDARY cannot be NONE");
    return false;
  }
  if (optionsCalc.boundaryRight == FLB::TypesNodes::NONE) 
  {
    terminal -> printf("[CALCULATION OPTIONS ERROR] The boundary condition RIGHT_BOUNDARY cannot be NONE");
    return false;
  }
  if (optionsCalc.boundaryUp == FLB::TypesNodes::NONE) 
  {
    terminal -> printf("[CALCULATION OPTIONS ERROR] The boundary condition UP_BOUNDARY cannot be NONE");
    return false;
  }
  if (optionsCalc.boundaryDown == FLB::TypesNodes::NONE) 
  {
    terminal -> printf("[CALCULATION OPTIONS ERROR] The boundary condition DOWN_BOUNDARY cannot be NONE");
    return false;
  }

  // Cuda check
  if (checkAllOptions && optionsCalc.calculationAPI == CalculationAPI::CUDA)
  {
    if (optionsCalc.cudaBlockSize.x <= 0)
    {
      terminal -> printf("[CALCULATION OPTIONS ERROR] The value of the cuda block size in the x direction cannot be negative or equal to zero\n");
      return false;
    }

    if (optionsCalc.cudaBlockSize.y <= 0)
    {
      terminal -> printf("[CALCULATION OPTIONS ERROR] The value of the cuda block size in the y direction cannot be negative or equal to zero\n");
      return false;
    }

    if (optionsCalc.typeAnalysis == 1 && optionsCalc.cudaBlockSize.z <= 0)
    {
      terminal -> printf("[CALCULATION OPTIONS ERROR] The value of the cuda block size in the z direction cannot be negative or equal to zero for a 3D analysis\n");
      return false;
    }

    uint32_t cudaMaxThreadsPerBlock = FLB::CudaUtils::getMaxThreadsPerBlock();
    uint32_t numberThreadsPerBlock2D = optionsCalc.cudaBlockSize.x * optionsCalc.cudaBlockSize.y;
    uint32_t numberThreadsPerBlock3D = optionsCalc.cudaBlockSize.x * optionsCalc.cudaBlockSize.y * optionsCalc.cudaBlockSize.z;
    if (optionsCalc.typeAnalysis == 0 && numberThreadsPerBlock2D > cudaMaxThreadsPerBlock)
    {
      terminal -> printf("[CALCULATION OPTIONS ERROR] The threads per block is superior to the maximum, %d > %d\n", numberThreadsPerBlock2D, cudaMaxThreadsPerBlock);
      return false;
    }
    else if (optionsCalc.typeAnalysis == 1 && numberThreadsPerBlock3D > cudaMaxThreadsPerBlock)
    {
      terminal -> printf("[CALCULATION OPTIONS ERROR] The threads per block is superior to the maximum, %d > %d\n", numberThreadsPerBlock3D, cudaMaxThreadsPerBlock);
      return false;
    }  
  }

  // check options only when there are not graphics
  if (checkAllOptions && optionsCalc.graphicsAPI == FLB::API::NONE)
  {
    if (FLB::Math::essentiallyEqual<float>(optionsCalc.timeSave, 0.0f, 1e-6))
    {
	terminal -> printf("[CALCULATION OPTIONS ERROR] The save time cannot be 0\n");
      return false;
    }
    
    if (FLB::Math::essentiallyEqual<float>(optionsCalc.timeSimulation, 0.0f, 1e-6))
    {
	terminal -> printf("[CALCULATION OPTIONS ERROR] The simulation time cannot be 0\n");
      return false;
    }

    if (optionsCalc.timeSave >= optionsCalc.timeSimulation)
    {
	terminal -> printf("[CALCULATION OPTIONS ERROR] The save time is greater than the simulation time, %6.4f > %6.4f\n", optionsCalc.timeSave, optionsCalc.timeSimulation);
      return false;
    }
  }

  return true;
}

bool FLB::CalculationReader::getOptionCalculation(std::string& option, std::string& value, FLB::OptionsCalculation& optionsCalc, unsigned int posLine, Fl_Simple_Terminal* terminal)
{
  bool errorOption = false;

  if (option == "TYPE_ANALYSIS") 
  {
    if (value == "2D") {optionsCalc.typeAnalysis = 0;}
    else if (value == "3D") {optionsCalc.typeAnalysis = 1;}
    else {errorOption = true;}
  }
  
  else if (option == "CALCULATION_API") 
  {
    if (value == "CUDA") {optionsCalc.calculationAPI = FLB::CalculationAPI::CUDA;}
    else {errorOption = true;}
  }

  // this option is only obligatory when CUDA is goint to be used 
  else if (option == "CUDA_BLOCK_SIZE")
  {
    std::regex r("\\s+"); // Delete whitespaces
    value = std::regex_replace(value, r, "");
    optionsCalc.cudaBlockSize.x = std::stoi(FLB::GeometryReader::extractString(value, '(', ',', posLine));
    optionsCalc.cudaBlockSize.y = std::stoi(FLB::GeometryReader::extractString(value, ',', ',', 1, 2, posLine));
    optionsCalc.cudaBlockSize.z = std::stoi(FLB::GeometryReader::extractString(value, ',', ')', 2, 1, posLine));
  }

  else if (option == "GRAPHICS_API") 
  {
    if (value == "NONE") {optionsCalc.graphicsAPI = FLB::API::NONE;}
    else if (value == "OPENGL") {optionsCalc.graphicsAPI = FLB::API::OPENGL;}
    else {errorOption = true;}
  }

  else if (option == "TYPE_PROBLEM")
  {
    if (value == "MONOFLUID") {optionsCalc.typeProblem = TypeProblem::MONOFLUID;} 
    else if (value == "FREE_SURFACE") {optionsCalc.typeProblem = TypeProblem::FREE_SURFACE;}
    else {errorOption = true;}
  }
  
  else if (option == "GRAVITY")
  {
    if (value == "FALSE") {optionsCalc.useGravity = false;} 
    else if (value == "TRUE") {optionsCalc.useGravity = true;}
    else {errorOption = true;}
  }

  else if (option == "SURFACE_TENSION")
  {
    if (value == "FALSE") {optionsCalc.surfaceTension = false;} 
    else if (value == "TRUE") {optionsCalc.surfaceTension = true;}
    else {errorOption = true;}
  }

  else if (option == "PRECISION") 
  {
    if (value == "16") {optionsCalc.precision = 16;}
    else if (value == "32") {optionsCalc.precision = 32;}
    else if (value == "64") {optionsCalc.precision = 64;}
    else {errorOption = true;} 
  }

  else if (option == "FLOW") 
  {
    try
    {
      optionsCalc.flow = std::stod(value);
    }
    catch(...)
    {
      terminal -> printf("[CALCULATION OPTIONS ERROR] The value of the flow in the calculation's file in the line %d is invalid\n", posLine);
      return false;
    }
    if (optionsCalc.flow < 0)
    {
      terminal -> printf("[CALCULATION OPTIONS ERROR] The value of the flow in the calculation's file in the line %d cannot be negative\n", posLine);
      return false;
    }
  }
  
  else if (option == "VELOCITY") 
  {
    try
    {
      optionsCalc.velocity = std::stod(value);
    }
    catch(...)
    {
      terminal -> printf("[CALCULATION OPTIONS ERROR] The value of the velocity in the calculation's file in the line %d is invalid\n", posLine);
      return false;
    }
    if (optionsCalc.flow < 0)
    {
      terminal -> printf("[CALCULATION OPTIONS ERROR] The value of the velocity in the calculation's file in the line %d cannot be negative\n", posLine);
      return false;
    }
  }
  
  else if (option == "KINEMATIC_VISCOSITY") 
  {
    try
    {
      optionsCalc.kinematicViscosity = std::stod(value);
    }
    catch(...)
    {
      terminal -> printf("[CALCULATION OPTIONS ERROR] The value of the kinematic viscosity in the calculation's file in the line %d is invalid\n", posLine);
      return false;
    }
    if (optionsCalc.kinematicViscosity < 0)
    {
      terminal -> printf("[CALCULATION OPTIONS ERROR] The value of the kinematic viscosity in the calculation's file in the line %d cannot be negative\n", posLine);
      return false;
    }
  }
  
  else if (option == "DENSITY") 
  {
    try
    {
      optionsCalc.density = std::stod(value);
    }
    catch(...)
    {
      terminal -> printf("[CALCULATION OPTIONS ERROR] The value of the density in the calculation's file in the line %d is invalid\n", posLine);
      return false;
    }
    if (optionsCalc.kinematicViscosity < 0)
    {
      terminal -> printf("[CALCULATION OPTIONS ERROR] The value of the density in the calculation's file in the line %d cannot be negative\n", posLine);
      return false;
    }
  }

  else if (option == "LEFT_BOUNDARY")
  {
    getTypeNode(optionsCalc.boundaryLeft, value);
  }

  else if (option == "RIGHT_BOUNDARY")
  {
    getTypeNode(optionsCalc.boundaryRight, value);
  }

  else if (option == "UP_BOUNDARY")
  {
    getTypeNode(optionsCalc.boundaryUp, value);
  }
  
  else if (option == "DOWN_BOUNDARY")
  {
    getTypeNode(optionsCalc.boundaryDown, value);
  }
  
  else if (option == "TIME_SIMULATION") 
  {
    try
    {
      optionsCalc.timeSimulation = std::stof(value);
    }
    catch(...)
    {
      terminal -> printf("[CALCULATION OPTIONS ERROR] The value of the time in the calculation's file in the line %d is invalid\n", posLine);
      return false;
    }
    if (optionsCalc.timeSimulation < 0)
    {
      terminal -> printf("[CALCULATION OPTIONS ERROR] The value of the time in the calculation's file in the line %d cannot be negative\n", posLine);
      return false;
    }
  }
  
  else if (option == "TIME_SAVE") 
  {
    try
    {
      optionsCalc.timeSave = std::stof(value);
    }
    catch(...)
    {
      terminal -> printf("[CALCULATION OPTIONS ERROR] The value of the time in the calculation's file in the line %d is invalid\n", posLine);
      return false;
    }
    if (optionsCalc.timeSave < 0)
    {
      terminal -> printf("[CALCULATION OPTIONS ERROR] The value of the time in the calculation's file in the line %d cannot be negative\n", posLine);
      return false;
    }
  }
  
  else 
  {
    terminal -> printf("[CALCULATION OPTIONS ERROR] The option in the calculation's file in the line %d doesn't exist\n", posLine);
    return false;
  }

  if (errorOption) 
  {
    terminal -> printf("[CALCULATION OPTIONS ERROR] The value of the option in the calculation's file in the line %d doesn't exist\n", posLine);
    return false;
  } 
  return true;
}

void FLB::CalculationReader::getTypeNode(FLB::TypesNodes& typeNode, std::string& value)
{
  if (value == "INPUT") typeNode = FLB::TypesNodes::INLET;
  if (value == "OUTPUT") typeNode = FLB::TypesNodes::OUTLET;
  if (value == "WALL") typeNode = FLB::TypesNodes::WALL;
}

bool FLB::CalculationReader::readOptionsCalculation(std::string& filePath, FLB::OptionsCalculation& optionsCalc, Fl_Simple_Terminal* terminal)
{
  unsigned int posLine = 0;
  std::string textFile;
  std::ifstream fileIn(filePath);
  if (fileIn.is_open())
  {
    while (std::getline(fileIn, textFile))
    {
      posLine += 1;
      // create copy and delete whitespaces
      std::string removedWhiteSpaces = textFile;
      removedWhiteSpaces.erase(std::remove(removedWhiteSpaces.begin(), removedWhiteSpaces.end(), ' '), removedWhiteSpaces.end());
      if (removedWhiteSpaces[0] == '#' || textFile.empty()) continue;
      
      std::string optionCalc[2];
      if(!splitString(optionCalc, textFile, '=', posLine, terminal, 1)) {fileIn.close(); return false;}
      if (!getOptionCalculation(optionCalc[0], optionCalc[1], optionsCalc, posLine, terminal)) {fileIn.close(); return false;}
    }

    if (!checkOptionsCalculation(optionsCalc, terminal)) {fileIn.close(); return false;}
  }
  fileIn.close();
  return true;
}

bool FLB::CalculationReader::readSomeOptionsCalculation(std::string &filePath, FLB::OptionsCalculation &optionsCalc, Fl_Simple_Terminal *terminal, std::vector<std::string>& optionsToSearch)
{
  unsigned int posLine = 0;
  std::string textFile;
  std::ifstream fileIn(filePath);
  if (fileIn.is_open())
  {
    while (std::getline(fileIn, textFile))
    {
      posLine += 1;
      // Delete whitespaces
      textFile.erase(std::remove(textFile.begin(), textFile.end(), ' '), textFile.end());
      if (textFile[0] == '#' || textFile.empty()) continue;
      
      std::string optionCalc[2];
      if(!splitString(optionCalc, textFile, '=', posLine, terminal, 1)) {fileIn.close(); return false;}

      // continue onlye if the value is in the options to search
      if (std::find(optionsToSearch.begin(), optionsToSearch.end(), optionCalc[0]) == optionsToSearch.end()) continue;
      if (!getOptionCalculation(optionCalc[0], optionCalc[1], optionsCalc, posLine, terminal)) {fileIn.close(); return false;}
    }
    
    if (!checkOptionsCalculation(optionsCalc, terminal, false)) {fileIn.close(); return false;}
  }
  fileIn.close();
  return true;

}
////////////////////////////////////// GeometryReader //////////////////////////////////////

FLB::GeometryReader::GeometryReader(std::filesystem::path directoryPath)
  : m_DirectoryPath(directoryPath)
{

}

bool FLB::GeometryReader::checkOptionsGeometry(FLB::Mesh *mesh, Fl_Simple_Terminal* terminal)
{
  if (mesh -> getIdxInitPointCDW() >= mesh -> getIdxEndPointCDW()) 
  {
    terminal -> printf("[GEOMETRY OPTIONS ERROR] The numeration of the initial point of the geometry cannot be equal or greater than the end point\n");
    return false;
  }
  return true;
}

bool FLB::GeometryReader::getMeshElements(std::string& element, std::string& coordinates, unsigned int posLine, FLB::Mesh* mesh, Fl_Simple_Terminal* terminal)
{
  std::string typeElement = FLB::GeometryReader::extractString(element, '(', posLine);

  std::regex r("\\s+"); // Delete whitespaces
  coordinates = std::regex_replace(coordinates, r, "");

  if (typeElement == "POINT") 
  {
    // get number of the element
    int elementNumber = std::stoi(FLB::GeometryReader::extractString(element, '(', ')', posLine));

    int countPoinst = mesh -> getPoints().size();
    if (countPoinst >= elementNumber)
    {
      terminal -> printf("[GEOMETRY OPTIONS ERROR] The numeration of the point in the line %d is not correct, maybe it is repeated\n", posLine);
      return false;
    }

    // get the coordinates of the point
    double x, y, z;
    x = std::stod(FLB::GeometryReader::extractString(coordinates, '(', ',', posLine));
    y = std::stod(FLB::GeometryReader::extractString(coordinates, ',', ',', 1, 2, posLine));
    z = std::stod(FLB::GeometryReader::extractString(coordinates, ',', ')', 2, 1, posLine));

    // Extract max and min values of the coordinates for the mesh
    if (x > mesh -> getxMax()) mesh -> getxMax() = x;
    if (x < mesh -> getxMin()) mesh -> getxMin() = x;
    if (y > mesh -> getyMax()) mesh -> getyMax() = y;
    if (y < mesh -> getyMin()) mesh -> getyMin() = y;
    if (z > mesh -> getzMax()) mesh -> getzMax() = z;
    if (z < mesh -> getzMin()) mesh -> getzMin() = z;

    // save point in mesh
    FLB::Point point = {x, y, z};
    mesh -> getPoints().push_back(point);
  }
  if (typeElement == "SIZE") mesh -> getSizeInterval() = std::stod(coordinates);
  if (typeElement == "TYPE_DOMAIN") 
  {
    if (coordinates == "2D") mesh -> getIs3D() = false;
    else if (coordinates == "3D") mesh -> getIs3D() = true;
  }
  return true;
}

bool FLB::GeometryReader::getCDWData(const std::string& option, const std::string& value, unsigned int posLine, FLB::Mesh* mesh, Fl_Simple_Terminal* terminal)
{
  bool errorOption = false;

  std::string description = extractString(option, '(', posLine);
  if (description == "INITIAL_POSITION_SHAPES")
  {
    try
    {
      mesh -> getIdxInitPointCDW() = std::stoi(value) - 1; // index in arrays start at 0
    }
    catch(...)
    {
      terminal -> printf("[GEOMETRY OPTIONS ERROR] The numeration of the point in the line %d is invalid\n", posLine);
      return false;
    }
    int countPoinst = mesh -> getPoints().size();
    if (mesh -> getIdxInitPointCDW() >= countPoinst) 
    { 
      terminal -> printf("[GEOMETRY OPTIONS ERROR] The numeration of the point in the line %d doesn't exist or it can't be the last point\n", posLine);
      return false;
    }
    
    if (mesh -> getIdxInitPointCDW() < 0) 
    {
      terminal -> printf("[GEOMETRY OPTIONS ERROR] The numeration of the point in the line %d can't be lower or equal to 0\n", posLine);
      return false;
    }
  }

  else if (description == "END_POSITION_SHAPES")
  {
    try
    {
      mesh -> getIdxEndPointCDW() = std::stoi(value) - 1; // index in arrays start at 0
    }
    catch(...)
    {
      terminal -> printf("[GEOMETRY OPTIONS ERROR] The numeration of the point in the line %d is invalid\n", posLine);
      return false;
    }
    int countPoinst = mesh -> getPoints().size();
    if (mesh -> getIdxEndPointCDW() > countPoinst)
    {
      terminal -> printf("[GEOMETRY OPTIONS ERROR] The numeration of the point in the line %d doesn't exist\n", posLine);
      return false;
    }
    if (mesh -> getIdxEndPointCDW() < 0)
    {
      terminal -> printf("[GEOMETRY OPTIONS ERROR] The numeration of the point in the line %d can't be lower or equal to 0\n", posLine);
      return false;
    }
  }
  
  else if (description == "CDW")
  {
    // get number of the element
    int elementNumber = std::stoi(FLB::GeometryReader::extractString(option, '(', ')', posLine));
    int countCDWs = mesh -> getCDWs().size();
    bool existCDW = elementNumber <= countCDWs;
    std::string attribute = FLB::GeometryReader::extractString(option, ')', posLine, true);
    if (attribute == "type") 
    {
      if (existCDW) 
      {
	terminal -> printf("[GEOMETRY OPTIONS ERROR] The numeration of the CDW in the line %d already exist\n", posLine);
	return false;
      }
      if (value == "CIRCLE" && !existCDW)
      {
	std::unique_ptr<FLB::CDW> circularCDW = std::make_unique<FLB::CircularCDW>();
	circularCDW -> typeCDW = FLB::TypesCDW::CIRCLE;
	mesh -> getCDWs().push_back(std::move(circularCDW));
      }
      else if (value == "RECTANGLE" && !existCDW)
      {
	std::unique_ptr<FLB::CDW> rectangularCDW = std::make_unique<FLB::RectangularCDW>();
	rectangularCDW -> typeCDW = FLB::TypesCDW::RECTANGLE;
	mesh -> getCDWs().push_back(std::move(rectangularCDW));
      }
      else 
      {
	terminal -> printf("[GEOMETRY OPTIONS ERROR] The shape of the CDW in the line %d is not correct\n", posLine);
	return false;
      }
    }

    else if (attribute == "height")
    {
      if (!existCDW)
      {
	terminal -> printf("Error in the line %d. It is neccesary to specify the type of CDW first\n", posLine);
	return false;
      }
      if (mesh -> getCDWs()[elementNumber - 1] -> typeCDW == TypesCDW::RECTANGLE) mesh -> getCDWs()[elementNumber - 1] -> setValue(std::stod(value), 0);
    }

    else if (attribute == "width")
    {
      if (!existCDW) 
      {
	terminal -> printf("Error in the line %d. It is neccesary to specify the type of CDW first\n", posLine);
	return false;
      }
      if (mesh -> getCDWs()[elementNumber - 1] -> typeCDW == TypesCDW::RECTANGLE) mesh -> getCDWs()[elementNumber -1] -> setValue(std::stod(value), 1);
	//mesh -> CDWs[elementNumber - 1].width = std::stod(value);
    }

    else if (attribute == "radius")
    {
      if (!existCDW)
      {
	terminal -> printf("Error in the line %d. It is neccesary to specify the type of CDW first\n", posLine);
	return false;
      }
      if (mesh -> getCDWs()[elementNumber - 1] -> typeCDW == TypesCDW::CIRCLE) mesh -> getCDWs()[elementNumber - 1] -> setValue(std::stod(value));
    }
  }

  else if (description == "SEPARATION")
  {

  }

  return true;

}

bool FLB::GeometryReader::getObstaclesData(const std::string& option, const std::string& value, unsigned int posLine, FLB::Mesh* mesh, Fl_Simple_Terminal* terminal)
{
  bool errorOption = false;
  std::string description = extractString(option, '(', posLine);

  if (description == "OBSTACLE")
  {
    // get number of the element
    int elementNumber = std::stoi(FLB::GeometryReader::extractString(option, '(', ')', posLine));
    int countObstacles = mesh -> getObstacles().size();
    bool existObstacle = elementNumber <= countObstacles;
    std::string attribute = FLB::GeometryReader::extractString(option, ')', posLine, true);
    if (attribute == "type")
    {
      if (existObstacle) 
      {
	terminal -> printf("[GEOMETRY OPTIONS ERROR] The numeration of the CDW in the line %d already exist\n", posLine);
	return false;
      }
      if (value == "WALL")
      {
	std::unique_ptr<FLB::Shape> rectangleShape = std::make_unique<FLB::RectangleShape>();
	rectangleShape -> typeShape = FLB::TypeShape::Rectangle;
	mesh -> getObstacles().push_back(std::move(rectangleShape));
      }
      if  (value == "CIRCLE")
      {
	std::unique_ptr<FLB::Shape> circleShape = std::make_unique<FLB::CircleShape>();
	circleShape -> typeShape = FLB::TypeShape::Circle;
	mesh -> getObstacles().push_back(std::move(circleShape));
      }
      if (value == "CSV_2D")
      {
	std::unique_ptr<FLB::Shape> importedShape = std::make_unique<FLB::Imported2DShape>();
	importedShape -> typeShape = FLB::TypeShape::Imported2DShape;
	mesh -> getObstacles().push_back(std::move(importedShape));
      }
    }
    
    else if (attribute == "name")
    {
      if (!existObstacle)
      {
	terminal -> printf("Error in the line %d. It is neccesary to specify the type of Obstacle first\n", posLine);
	return false;
      }

      if (mesh -> getObstacles()[elementNumber - 1] -> typeShape == FLB::TypeShape::Imported2DShape)
      {
	std::string filename = value + ".csv";
	mesh -> getObstacles()[elementNumber -1] -> setFilename(filename);
      }
    }
    
    else if (attribute == "position")
    {
      if (!existObstacle)
      {
	terminal -> printf("Error in the line %d. It is neccesary to specify the type of Obstacle first\n", posLine);
	return false;
      }
      // get the coordinates of the center
      double x, y, z;
      x = std::stod(FLB::GeometryReader::extractString(value, '(', ',', posLine));
      y = std::stod(FLB::GeometryReader::extractString(value, ',', ',', 1, 2, posLine));
      z = std::stod(FLB::GeometryReader::extractString(value, ',', ')', 2, 1, posLine));
     
      FLB::TypeShape type = mesh -> getObstacles()[elementNumber - 1] -> typeShape; 
      if (type == FLB::TypeShape::Rectangle || type == FLB::TypeShape::Circle || type == FLB::TypeShape::Imported2DShape)
      {
	mesh -> getObstacles()[elementNumber -1] -> setValue(x, FLB::TypeValue::x0);
	mesh -> getObstacles()[elementNumber -1] -> setValue(y, FLB::TypeValue::y0);
	mesh -> getObstacles()[elementNumber -1] -> setValue(z, FLB::TypeValue::z0);
      }
    }
    
    else if (attribute == "rotation")
    {
      if (!existObstacle)
      {
	terminal -> printf("Error in the line %d. It is neccesary to specify the type of Obstacle first\n", posLine);
	return false;
      }
      // get the rotation in sexadecimal degrees
      double xRotation, yRotation, zRotation;
      xRotation = std::stod(FLB::GeometryReader::extractString(value, '(', ',', posLine));
      yRotation = std::stod(FLB::GeometryReader::extractString(value, ',', ',', 1, 2, posLine));
      zRotation = std::stod(FLB::GeometryReader::extractString(value, ',', ')', 2, 1, posLine));
    
      // transform to radians
      constexpr double pi = 3.141592653589793;
      xRotation = xRotation * pi / 180.0;
      yRotation = yRotation * pi / 180.0;
      zRotation = zRotation * pi / 180.0;
      FLB::TypeShape type = mesh -> getObstacles()[elementNumber - 1] -> typeShape; 
      if (type == FLB::TypeShape::Rectangle || type == FLB::TypeShape::Circle || type == FLB::TypeShape::Imported2DShape)
      {
	mesh -> getObstacles()[elementNumber -1] -> setValue(xRotation, FLB::TypeValue::xRotation);
	mesh -> getObstacles()[elementNumber -1] -> setValue(yRotation, FLB::TypeValue::yRotation);
	mesh -> getObstacles()[elementNumber -1] -> setValue(zRotation, FLB::TypeValue::zRotation);
      }
    }
    
    else if (attribute == "height")
    {
      if (!existObstacle)
      {
	terminal -> printf("Error in the line %d. It is neccesary to specify the type of Obstacle first\n", posLine);
	return false;
      }
      if (mesh -> getObstacles()[elementNumber - 1] -> typeShape == FLB::TypeShape::Rectangle) mesh -> getObstacles()[elementNumber -1] -> setValue(std::stod(value), FLB::TypeValue::Height);
    }
    
    else if (attribute == "xWidth")
    {
      if (!existObstacle)
      {
	terminal -> printf("Error in the line %d. It is neccesary to specify the type of Obstacle first\n", posLine);
	return false;
      }
      if (mesh -> getObstacles()[elementNumber - 1] -> typeShape == FLB::TypeShape::Rectangle) mesh -> getObstacles()[elementNumber -1] -> setValue(std::stod(value), FLB::TypeValue::xWidth);
    }
    
    else if (attribute == "zWidth")
    {
      if (!existObstacle)
      {
	terminal -> printf("Error in the line %d. It is neccesary to specify the type of Obstacle first\n", posLine);
	return false;
      }
      if (mesh -> getObstacles()[elementNumber - 1] -> typeShape == FLB::TypeShape::Rectangle) mesh -> getObstacles()[elementNumber -1] -> setValue(std::stod(value), FLB::TypeValue::zWidth);

    }
    
    else if (attribute == "radius")
    {
      if (!existObstacle)
      {
	terminal -> printf("Error in the line %d. It is neccesary to specify the type of Obstacle first\n", posLine);
	return false;
      }
      if (mesh -> getObstacles()[elementNumber - 1] -> typeShape == FLB::TypeShape::Circle) mesh -> getObstacles()[elementNumber -1] -> setValue(std::stod(value), FLB::TypeValue::Radius);
    }
    
    else if (attribute == "thickness")
    {
      if (!existObstacle)
      {
	terminal -> printf("Error in the line %d. It is neccesary to specify the type of Obstacle first\n", posLine);
	return false;
      }
      if (mesh -> getObstacles()[elementNumber - 1] -> typeShape == FLB::TypeShape::Circle) mesh -> getObstacles()[elementNumber -1] -> setValue(std::stod(value), FLB::TypeValue::Thickness);
    }

  }

  return true;
}

bool FLB::GeometryReader::modifyGeometryData(FLB::Mesh* mesh, Fl_Simple_Terminal* terminal)
{
  auto& points = mesh -> getPoints();
  double heightInitPointCDW = points[mesh -> getIdxInitPointCDW()].y;
  double heightEndPointCDW = points[mesh -> getIdxEndPointCDW()].y;
  double widthInitPointCDW = points[mesh -> getIdxInitPointCDW()].z;
  double widthtEndPointCDW = points[mesh -> getIdxEndPointCDW()].z;
  double initAuxHeight = 0.0f, endAuxHeight = 0.0f, initAuxWidth = 0.0f, endAuxWidth = 0.0f;
  int count = 0;
  // For all the CDWs it is necesary to check if the coordinates of the poinst in its length plus its height and width are greater that the max height and width of the domain
  auto& CDWs =  mesh -> getCDWs();
  for(std::unique_ptr<FLB::CDW>& cdw : CDWs)
  {
    if (cdw -> typeCDW == FLB::TypesCDW::CIRCLE)
    {
      initAuxHeight = heightInitPointCDW + 2 * cdw -> getValue(); //sum of diameter
      endAuxHeight = heightEndPointCDW + 2 * cdw -> getValue();
      // Only extremes should be taken into account because the rest is due to the separation between CDWs
      if (count == 0 || count == (CDWs.size() - 1)) initAuxWidth += cdw -> getValue(); // sum of radius
    }
    else if (cdw -> typeCDW == FLB::TypesCDW::RECTANGLE)
    {
      initAuxHeight = heightInitPointCDW + cdw -> getValue(0); //sum of the height
      endAuxHeight = heightEndPointCDW + cdw -> getValue(0);
      // Only extremes should be taken into account because the rest is due to the separation between CDWs
      if (count == 0 || count == (CDWs.size() - 1)) initAuxWidth += 0.5f * cdw -> getValue(1); // sum the half of the width
    }
    if (initAuxHeight > mesh -> getyMax()) mesh -> getyMax() = initAuxHeight;
    if (endAuxHeight > mesh -> getyMax()) mesh -> getyMax() = endAuxHeight;
    count += 1;
  }
 
  // For all the imported shapes and with all the information they needed already read, its initialization is now posible
  std::vector<std::unique_ptr<FLB::Shape>>& obstacles = mesh -> getObstacles();
 
  for (std::unique_ptr<FLB::Shape>& obstacle : obstacles)
  {
    // set if the domain is 3D in the obstacles
    obstacle -> setIs3D(mesh -> is3D());
    FLB::TypeShape type = obstacle -> typeShape;
    if (type == FLB::TypeShape::Imported2DShape || type == FLB::TypeShape::Rectangle) 
    {
      // return if some error is produced
      if (!obstacle -> initShape(m_DirectoryPath, terminal)) return false;
    }
  }

  // set if the domain is 3D in the CDW
  const std::vector<std::unique_ptr<FLB::CDW>>& cdws = mesh -> getCDWs(); //cross drainage works
  for (const std::unique_ptr<FLB::CDW>&  cdw : cdws) 
  {
    cdw -> setIs3D(mesh -> is3D());
  }
  
  // Only if the domain is 3D, the domain will have width 
  if (mesh -> is3D())
  {
    // if there is only one CDW, its local width has to be double because there is not separation
    if (CDWs.size() == 1) initAuxWidth = widthInitPointCDW + 2 * initAuxWidth;
    else initAuxWidth += widthInitPointCDW + (CDWs.size() - 1) * mesh -> getSeparationCDW();
    endAuxWidth += initAuxWidth - widthInitPointCDW + widthtEndPointCDW;

    if (initAuxWidth > mesh -> getzMax()) mesh -> getzMax() = initAuxWidth;
    if (endAuxWidth > mesh -> getzMax()) mesh -> getzMax() = endAuxWidth;
  }
  return true;
}

bool FLB::GeometryReader::readGeometryOptions(std::string filePath, FLB::Mesh* mesh, Fl_Simple_Terminal* terminal)
{
  unsigned int posLine = 0; //line to know where the posible error is located
  std::string textFile;
  std::ifstream fileIn(filePath);
  if (fileIn.is_open())
  {
    while (std::getline(fileIn, textFile))
    {
      posLine += 1;
      // create copy and delete whitespaces
      std::string removedWhiteSpaces = textFile;
      removedWhiteSpaces.erase(std::remove(removedWhiteSpaces.begin(), removedWhiteSpaces.end(), ' '), removedWhiteSpaces.end());
      if (removedWhiteSpaces[0] == '#' || textFile.empty()) continue;

      std::string geometryOption[2];
      if (!splitString(geometryOption, textFile, '=', posLine, terminal, 0)) {fileIn.close(); return false;}
      
      if (!getMeshElements(geometryOption[0], geometryOption[1], posLine, mesh, terminal)) {fileIn.close(); return false;}

      if (!getCDWData(geometryOption[0], geometryOption[1], posLine, mesh, terminal)) {fileIn.close(); return false;}

      if (!getObstaclesData(geometryOption[0], geometryOption[1], posLine, mesh, terminal)) {fileIn.close(); return false;}
    }
    if (!modifyGeometryData(mesh, terminal)) {fileIn.close(); return false;};
    if (!checkOptionsGeometry(mesh, terminal)) {fileIn.close(); return false;}

  }
  fileIn.close();
  return true;
}
