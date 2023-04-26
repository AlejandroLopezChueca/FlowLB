#pragma once
#include "io/reader.h"

namespace FLB 
{
  class ErrorHandler
  {
    public:
      ErrorHandler();
      ~ErrorHandler();

      void catchErrorConfiguration();

  };

}
