#include "reader.h"
#include <array>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <iterator>
#include <ostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <algorithm>

void FLB::Reader::splitString(std::string optionCalc[], std::string textFile, char delimiter, int posLine)
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

void FLB::CalculationReader::getOptionCalculation(std::string option, std::string value, FLB::OptionsCalculation& OptionsCalc, int posLine)
{
  bool errorOption = false;

  if (option == "TYPE_ANALYSIS") 
  {
    if (value == "2D") {OptionsCalc.typeAnalysis = 0;}
    else if (value == "3D") {OptionsCalc.typeAnalysis = 1;}
    else {errorOption = true;}
  }
  else if (option == "TIME_ANALYSIS") 
  {
    if (value == "TRUE") {OptionsCalc.timeAnalysis = true;}
    else if (value == "FALSE") {OptionsCalc.timeAnalysis = false;}
    else {errorOption = true;}
  }
  else if (option == "PLOT_GRAPHICS") 
  {
    if (value == "TRUE") {OptionsCalc.plotGraphics = true;}
    else if (value == "FALSE") {OptionsCalc.plotGraphics = false;}
    else {errorOption = true;}
  }
  else if (option == "GRAPHICS_API") 
  {
    if (value == "NONE") {OptionsCalc.graphicsAPI = RendererAPI::NONE;}
    else if (value == "OPENGL") {OptionsCalc.graphicsAPI = RendererAPI::OPENGL;}
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
  else if (option == "CONVERGENCE_ERROR") 
  {
    try
    {
      OptionsCalc.maxError = std::stof(value);
    }
    catch(...)
    {
      throw std::invalid_argument("The value of the convergence error in the calculation's file in the line " + std::to_string(posLine) + " is invalid");
    } 
  }
  else {throw std::out_of_range("The option in the calculation's file in the line " + std::to_string(posLine) + " doesn't exist");}
  if (errorOption) {throw std::out_of_range("The value of the option in the calculation's file in the line " + std::to_string(posLine) + " doesn't exist");} 
}

void FLB::CalculationReader::readOptionsCalculation(std::string filePath, FLB::OptionsCalculation& OptionsCalc)
{
  int posLine = 1;
  std::string textFile;
  std::ifstream fileIn(filePath);
  if (fileIn.is_open())
  {
    while (std::getline(fileIn, textFile))
    {
      std::string optionCalc[2];
      if (textFile[0] == '#' || textFile.empty()){continue;}
      else
      {
	FLB::CalculationReader::splitString(optionCalc, textFile, '=', posLine);
	FLB::CalculationReader::getOptionCalculation(optionCalc[0], optionCalc[1], OptionsCalc, posLine);
      }
      posLine += 1;
    }
  }
  fileIn.close();
}
