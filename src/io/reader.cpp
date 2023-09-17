#include <array>
#include <cstddef>
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


#include "geometry/mesh.h"
#include "geometry/shapes.h"
#include "reader.h"

void FLB::Reader::splitString(std::string optionCalc[], std::string textFile, char delimiter, unsigned int posLine)
{
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
      throw std::out_of_range("Error in the line " + std::to_string(posLine));
    }
    
    text = std::regex_replace(text, r, "");
    optionCalc[i] = text;  
    i += 1;
  }
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
    throw std::out_of_range("The option in the geometry's file in the line " + std::to_string(posLine) + " is incorrect");
  }
  return value;

}

void FLB::CalculationReader::checkOptionsCalculation(FLB::OptionsCalculation& optionsCalc)
{
  if (optionsCalc.flow < 1e-5)
  {
    throw std::invalid_argument("The flow argument has to have a value");
  }
}

void FLB::CalculationReader::getOptionCalculation(std::string option, std::string value, FLB::OptionsCalculation& OptionsCalc, unsigned int posLine)
{
  bool errorOption = false;

  if (option == "TYPE_ANALYSIS") 
  {
    if (value == "2D") {OptionsCalc.typeAnalysis = 0;}
    else if (value == "3D") {OptionsCalc.typeAnalysis = 1;}
    else {errorOption = true;}
  }
  else if (option == "GRAPHICS_API") 
  {
    if (value == "NONE") {OptionsCalc.graphicsAPI = FLB::API::NONE;}
    else if (value == "OPENGL") {OptionsCalc.graphicsAPI = FLB::API::OPENGL;}
    else {errorOption = true;}
  }
  else if (option == "PRECISION") 
  {
    if (value == "16") {OptionsCalc.precision = 16;}
    else if (value == "32") { OptionsCalc.precision = 32;}
    else if (value == "64") {OptionsCalc.precision = 64;}
    else {errorOption = true;} 
  }
  else if (option == "FLOW") 
  {
    try
    {
      OptionsCalc.flow = std::stof(value);
    }
    catch(...)
    {
      throw std::invalid_argument("The value of the flow in the calculation's file in the line " + std::to_string(posLine) + " is invalid");
    } 
  }
  else {throw std::out_of_range("The option in the calculation's file in the line " + std::to_string(posLine) + " doesn't exist");}
  if (errorOption) {throw std::out_of_range("The value of the option in the calculation's file in the line " + std::to_string(posLine) + " doesn't exist");} 
}

void FLB::CalculationReader::readOptionsCalculation(std::string filePath, FLB::OptionsCalculation& optionsCalc)
{
  unsigned int posLine = 0;
  std::string textFile;
  std::ifstream fileIn(filePath);
  if (fileIn.is_open())
  {
    while (std::getline(fileIn, textFile))
    {
      posLine += 1;
      std::cout<<posLine<<std::endl;
      std::cout<<textFile.empty()<<"\n"<<std::endl;
      if (textFile[0] == '#' || textFile.empty()) continue;
      
      std::string optionCalc[2];
      FLB::CalculationReader::splitString(optionCalc, textFile, '=', posLine);
      FLB::CalculationReader::getOptionCalculation(optionCalc[0], optionCalc[1], optionsCalc, posLine);
    }

    checkOptionsCalculation(optionsCalc);
  }
  fileIn.close();
}

void FLB::GeometryReader::checkOptionsGeometry(FLB::Mesh *mesh)
{
  if (mesh -> initPointCDW >= mesh -> endPointCDW) throw std::invalid_argument("The numeration of the initial point of the geometry cannot be equal or greater than the end point");

}

void FLB::GeometryReader::getMeshElements(std::string element, std::string coordinates, unsigned int posLine, FLB::Mesh* mesh)
{
  std::string typeElement = FLB::GeometryReader::extractString(element, '(', posLine);

  std::regex r("\\s+"); // Delete whitespaces
  coordinates = std::regex_replace(coordinates, r, "");

  if (typeElement == "POINT") 
  {
    // get number of the element
    int elementNumber = std::stoi(FLB::GeometryReader::extractString(element, '(', ')', posLine));

    int countPoinst = mesh -> points.size();
    if (countPoinst >= elementNumber) throw std::out_of_range("The numeration of the point in the line" + std::to_string(posLine) + " is not correct, maybe it is repeated");
    // get the coordinates of the point
    float x, y, z;
    x = std::stof(FLB::GeometryReader::extractString(coordinates, '(', ',', posLine));
    y = std::stof(FLB::GeometryReader::extractString(coordinates, ',', ',', 1, 2, posLine));
    z = std::stof(FLB::GeometryReader::extractString(coordinates, ',', ')', 2, 1, posLine));

    // Extract max and min values of the coordinates for the mesh
    if (x > mesh -> xMax) mesh -> xMax = x;
    if (x < mesh -> xMin) mesh -> xMin = x;
    if (y > mesh -> yMax) mesh -> yMax = y;
    if (y < mesh -> yMin) mesh -> yMin = y;
    if (z > mesh -> zMax) mesh -> zMax = z;
    if (z < mesh -> zMin) mesh -> zMin = z;

    // save point in mesh
    FLB::Point point = {x, y, z};
    mesh -> points.push_back(point);
  }
  if (typeElement == "SIZE") mesh -> sizeInterval = std::stof(coordinates);
  if (typeElement == "TYPE_DOMAIN") 
  {
    if (coordinates == "2D") mesh -> is3D = false;
    else if (coordinates == "3D") mesh -> is3D = true;
  }
}

void FLB::GeometryReader::getCDWData(std::string option, std::string value, unsigned int posLine, FLB::Mesh* mesh)
{
  bool errorOption = false;

  std::string description = extractString(option, '(', posLine);
  if (description == "INITIAL_POSITION_SHAPES")
  {
    try
    {
      mesh -> initPointCDW = std::stoi(value);
    }
    catch(...)
    {
      throw std::invalid_argument("The numeration of the point in the line " + std::to_string(posLine) + " is invalid");
    }
    int countPoinst = mesh -> points.size();
    if (mesh -> initPointCDW >= countPoinst) throw std::out_of_range("The numeration of the point in the line " + std::to_string(posLine) + " doesn't exist or it can't be the last point");
    if (mesh -> initPointCDW <= 0) throw std::out_of_range("The numeration of the point in the line " + std::to_string(posLine) + " can't be lower or equal to 0");
  }

  else if (description == "END_POSITION_SHAPES")
  {
    try
    {
      mesh -> endPointCDW = std::stoi(value);
    }
    catch(...)
    {
      throw std::invalid_argument("The numeration of the point in the line " + std::to_string(posLine) + " is invalid");
    }
    int countPoinst = mesh -> points.size();
    if (mesh -> endPointCDW > countPoinst) throw std::invalid_argument("The numeration of the point in the " + std::to_string(posLine) + " doesnt't exist");
    if (mesh -> endPointCDW <= 0) throw std::out_of_range("The numeration of the point in the line " + std::to_string(posLine) + " can't be lower or equal to 0");
  }
  
  else if (description == "CDW")
  {
    // get number of the element
    int elementNumber = std::stoi(FLB::GeometryReader::extractString(option, '(', ')', posLine));
    int countCDWs = mesh -> CDWs.size();
    bool existCDW = elementNumber <= countCDWs;
    std::string attribute = FLB::GeometryReader::extractString(option, ')', posLine, true);
    if (attribute == "type") 
    {
      if (existCDW) throw std::invalid_argument("The numeration of the CDW in the line " + std::to_string(posLine) + " already exist");
      if (value == "CIRCLE" && !existCDW)
      {
	std::unique_ptr<FLB::CircularCDW> circularCDW(new FLB::CircularCDW());
	circularCDW -> typeCDW = FLB::TypesCDW::CIRCLE;
	mesh -> CDWs.push_back(std::move(circularCDW));
      }
      else if (value == "RECTANGLE" && !existCDW)
      {
	std::unique_ptr<FLB::RectangularCDW> rectangularCDW(new FLB::RectangularCDW());
	rectangularCDW -> typeCDW = FLB::TypesCDW::RECTANGLE;
	mesh -> CDWs.push_back(std::move(rectangularCDW));
      }
      else throw std::invalid_argument("The shape of the CDW in the line " + std::to_string(posLine) + " is not correct");
    }

    else if (attribute == "height")
    {
      if (!existCDW) std::invalid_argument("Error in the line " + std::to_string(posLine) + ". It is neccesary to specify the type of CDW first");
      if (mesh -> CDWs[elementNumber - 1] -> typeCDW == TypesCDW::RECTANGLE) mesh -> CDWs[elementNumber - 1] -> setValue(std::stof(value), 0);
    }

    else if (attribute == "width")
    {
      if (!existCDW) std::invalid_argument("Error in the line " + std::to_string(posLine) + ". It is neccesary to specify the type of CDW first");
      if (mesh -> CDWs[elementNumber - 1] -> typeCDW == TypesCDW::RECTANGLE) mesh -> CDWs[elementNumber -1] -> setValue(std::stof(value), 1);
	//mesh -> CDWs[elementNumber - 1].width = std::stof(value);
    }

    else if (attribute == "radius")
    {
      if (!existCDW) std::invalid_argument("Error in the line " + std::to_string(posLine) + ". It is neccesary to specify the type of CDW first");
      if (mesh -> CDWs[elementNumber - 1] -> typeCDW == TypesCDW::CIRCLE) mesh -> CDWs[elementNumber - 1] -> setValue(std::stof(value));
    }
  }

  else if (description == "SEPARATION");

}

void FLB::GeometryReader::getObstaclesData(std::string option, std::string value, unsigned int posLine, FLB::Mesh* mesh)
{
  bool errorOption = false;

  //if (option == "")

}

void FLB::GeometryReader::modifyGeometryData(FLB::Mesh* mesh)
{
  float heightInitPointCDW = mesh -> points[mesh -> initPointCDW].y;
  float heightEndPointCDW = mesh -> points[mesh -> endPointCDW].y;
  float widthInitPointCDW = mesh -> points[mesh -> initPointCDW].z;
  float widthtEndPointCDW = mesh -> points[mesh -> endPointCDW].z;
  float initAuxHeight = 0.0f, endAuxHeight = 0.0f, initAuxWidth = 0.0f, endAuxWidth = 0.0f;
  int count = 0;
  // For all the CDWs it is necesary to check if the coordinates of the poinst in its length plus its height and width are greater that the max height and width of the domain
  for(std::unique_ptr<FLB::CDW>& cdw : mesh -> CDWs)
  {
    if (cdw -> typeCDW == FLB::TypesCDW::CIRCLE)
    {
      initAuxHeight = heightInitPointCDW + 2 * cdw -> getValue(); //sum of diameter
      endAuxHeight = heightEndPointCDW + 2 * cdw -> getValue();
      // Only extremes should be taken into account because the rest is due to the separation between CDWs
      if (count == 0 || count == (mesh -> CDWs.size() - 1)) initAuxWidth += cdw -> getValue(); // sum of radius
    }
    else if (cdw -> typeCDW == FLB::TypesCDW::RECTANGLE)
    {
      initAuxHeight = heightInitPointCDW + cdw -> getValue(0); //sum of the height
      endAuxHeight = heightEndPointCDW + cdw -> getValue(0);
      // Only extremes should be taken into account because the rest is due to the separation between CDWs
      if (count == 0 || count == (mesh -> CDWs.size() - 1)) initAuxWidth += 0.5f * cdw -> getValue(1); // sum the half of the width
    }
    if (initAuxHeight > mesh -> yMax) mesh -> yMax = initAuxHeight;
    if (endAuxHeight > mesh -> yMax) mesh -> yMax = endAuxHeight;
    count += 1;
  }
  // Only if the domain is 3D, the domain will have width 
  if (mesh -> is3D)
  {
    // if there is only one CDW, its local width has to be double because there is not separation
    if (mesh -> CDWs.size() == 1) initAuxWidth = widthInitPointCDW + 2 * initAuxWidth;
    else initAuxWidth += widthInitPointCDW + (mesh -> CDWs.size() - 1) * mesh -> separationCDW;
    endAuxWidth += initAuxWidth - widthInitPointCDW + widthtEndPointCDW;

    if (initAuxWidth > mesh -> zMax) mesh -> zMax = initAuxWidth;
    if (endAuxWidth > mesh -> zMax) mesh -> zMax = endAuxWidth;
  }
}

void FLB::GeometryReader::readGeometryOptions(std::string filePath, FLB::Mesh* mesh)
{
  unsigned int posLine = 0; //line to know where the posible error is located
  std::string textFile;
  std::ifstream fileIn(filePath);
  if (fileIn.is_open())
  {
    while (std::getline(fileIn, textFile))
    {
      posLine += 1;
      if (textFile[0] == '#' || textFile.empty()) continue;

      std::string geometryOption[2];
      splitString(geometryOption, textFile, '=', posLine);
      getMeshElements(geometryOption[0], geometryOption[1], posLine, mesh);
      getCDWData(geometryOption[0],geometryOption[1], posLine, mesh);
      getObstaclesData(geometryOption[0], geometryOption[1], posLine, mesh);
    }
    modifyGeometryData(mesh);
    checkOptionsGeometry(mesh);

  }
  fileIn.close();
}
