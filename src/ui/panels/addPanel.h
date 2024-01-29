#pragma once

#include "graphics/API.h"
#include "geometry/mesh.h"

#include <cstddef>

namespace FLB 
{
  class MetricsPanel
  {
    public:
      MetricsPanel(FLB::API, FLB::Mesh* mesh, float& time);
      ~MetricsPanel() = default;
      
      void onImGuiRender(bool* open);

    private:
      size_t m_NumPointsMesh;
      float& m_Time;


  };

}
