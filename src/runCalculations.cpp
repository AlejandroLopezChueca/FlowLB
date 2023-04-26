#include <iostream>
#include <ostream>
#include <string>
#include <cuda_runtime.h>

#include "io/reader.h"
#include "cuda/lbmKernel2D.cuh"
#include "runCalculations.h"
#include "graphics/renderer2D.h"

void FLB::runCalculations()
{
  //Default options
  FLB::OptionsCalculation OptionsCalc = 
  {
    0, //2D
    true,
    true,
    RendererAPI::NONE,
    32,
    0,
    0.001
  };
  
  FLB::CalculationReader CalcReader;
  std::string filePath = "/home/alejandro_lopez/Projects/FlowLB/test/OptionsCalc.txt";
  CalcReader.readOptionsCalculation(filePath, OptionsCalc);
  if (OptionsCalc.flow == 0)
  {
    throw std::invalid_argument("The flow argument has to have a value");
  }
  if (OptionsCalc.plotGraphics == true) 
  {
    OptionsCalc.timeAnalysis = false;
    Renderer2D Renderer;
  }
  else 
  {
    OptionsCalc.graphicsAPI = RendererAPI::NONE; 
  }
  FLB::h_runCalculationsGPU2D(OptionsCalc);

  //std::cout<<OptionsCalc.typeAnalysis<<std::endl;
  //std::cout<<OptionsCalc.plotGraphics<<std::endl;
  //std::cout<<OptionsCalc.precision<<std::endl;

}
