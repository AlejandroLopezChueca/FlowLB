#pragma once

#include "FL/Fl_Simple_Terminal.H"
#include "glm/fwd.hpp"
#include "graphics/API.h"
#include "geometry/mesh.h"
#include "graphics/cameraController.h"
#include "graphics/scene/components.h"
#include "graphics/scene/scene.h"
#include "graphics/texture.h"
#include "graphics/window.h"

#include <cstddef>
#include <cstdint>

namespace FLB 
{
  class MetricsPanel
  {
    public:
      MetricsPanel(FLB::API, FLB::Mesh* mesh, float& time, float& miliSecondsSimulation, float& frameRateSimulation);
      ~MetricsPanel() = default;
      
      void onImGuiRender(bool* open);
      void updateMemoryUsed(float freeMemory, float usedMemory) {m_FreeMemory = freeMemory; m_UsedMemory = usedMemory;}

    private:
      size_t m_NumPointsMesh;
      float& m_Time;  // time of simulation in seconds
      float& m_MiliSecondsSimulation;
      float& m_FrameRateSimulation;
      float m_FreeMemory = 0.0f; // Free memory GPU (MB)
      float m_UsedMemory = 0.0f; // Used memory GPU (MB)
    
      friend class RenderLayer;
  };

  class IsosurfacePanel
  {
    public:
      IsosurfacePanel(FLB::API api, FLB::CalculationAPI calculationApi, Fl_Simple_Terminal* terminal, const FLB::Mesh* mesh);

      void onImGuiRender(bool* open);
      void setScene(FLB::Scene* scene) {m_Scene = scene;}

    private:

      void createIsosurface();

      FLB::Scene* m_Scene;
      FLB::API m_Api;
      FLB::CalculationAPI m_CalculationApi;
      Fl_Simple_Terminal* m_Terminal;
      const FLB::Mesh* m_DomainMesh;

      // width and height of the texture
      int m_Width = 500;
      int m_Height = 500;
  };

  class ConsultValuesPanel
  {
    public:
      ConsultValuesPanel(const FLB::Texture2D* texture, const FLB::Window* window, const FLB::OrthographicCameraController* cameraController, const int& idxVectorRepresentation, const int& idScalarVectorField);

      void onImGuiRender(bool* open, const glm::vec2* viewPortBounds);

    private:
      void getValue(const glm::vec2* viewportBounds);

      const FLB::Texture2D* m_ScalarVectorialFieldsTexture;
      const FLB::Window* m_RenderWindow;
      const FLB::OrthographicCameraController* m_CameraController;
      float m_ConsultedValue = 0.0f;

      const int& m_CurrentIdxVectorRepresentation; // 0 = Magnitude, 1 = X component, 2 = Y component
      const int& m_IdScalarVectorField; // 0 for velocity, 1 for Pressure
  };

}
