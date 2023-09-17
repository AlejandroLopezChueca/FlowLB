#include <cstdint>
#include <iostream>
#include <ostream>
#include <string>
#include <cuda_runtime.h>
#include <filesystem>

#include "FL/Fl_Simple_Terminal.H"
#include "geometry/mesh.h"
#include "io/reader.h"
#include "mainProcesses.h"
#include "cuda/runCuda.cuh"
#include "initData.h"
#include "io/writer.h"

void FLB::runCalculations(const char* pathFiles, FLB::Mesh* mesh, Fl_Simple_Terminal* terminal)
{
  std::string geometryPath = pathFiles;
  std::string optionsCalculationPath = pathFiles;
  geometryPath += "/Geometry.txt";
  optionsCalculationPath += "/OptionsCalc.txt";
  if (!std::filesystem::exists(geometryPath)) terminal -> printf("The file Geometry.txt doesn't exist in %s\n", pathFiles);
  if (!std::filesystem::exists(optionsCalculationPath)) terminal -> printf("The file OptionsCalc.txt doesn't exist in %s\n", pathFiles);
  //Default options
  FLB::OptionsCalculation optionsCalc;
  
  FLB::CalculationReader CalcReader;
  CalcReader.readOptionsCalculation(optionsCalculationPath, optionsCalc);

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
    }
    //FLB::h_runCudaCalculations2D<float>(optionsCalc, h_weights, mesh -> numPointsMesh, maxIterations, numDimensions, numVelocities);
  }
  
  else if (optionsCalc.precision == 64)
  {

    double h_weights[9] = {4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};

  
    //FLB::h_runCudaCalculations2D<double>(optionsCalc, h_weights, mesh -> numPointsMesh, maxIterations, numDimensions, numVelocities);
  }

};


void FLB::createDomain(const char *pathFiles, FLB::Mesh* mesh, Fl_Simple_Terminal* terminal)
{
  std::string geometryPath = pathFiles;
  geometryPath += "/Geometry.txt";
  if (!std::filesystem::exists(geometryPath))
  {
    terminal -> printf("The file Geometry.txt doesn't exist in %s\n", pathFiles);
    return;
  }
  FLB::GeometryReader geometryReader;
  geometryReader.readGeometryOptions(geometryPath, mesh);
  // get number poinst mesh in each direction
  mesh -> getNumberPointsMesh();
  // Reserve memory for arrays of the mesh
  mesh -> reserveMemory();
  //Get the coordinates of each point of the mesh in Si units
  mesh -> getCoordinatesPoints();
  FLB::initData<float, FLB::voidInitVelocityAndf<float>>(9, mesh);
  // save mesh
  FLB::VTUWriter writer;
  std::string pathSave = pathFiles;
  pathSave += "/mesh.vtu";
  writer.addField<uint8_t>("UInt8", "Flags", mesh -> domainFlags, mesh -> numPointsMesh, true);
  writer.savePointData(pathSave, mesh, terminal); 
}
