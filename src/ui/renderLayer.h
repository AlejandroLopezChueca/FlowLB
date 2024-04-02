#pragma once

#include "graphics/API.h"
#include "graphics/cameraController.h"
#include "graphics/frameBuffer.h"
#include "graphics/scene/components.h"
#include "graphics/texture.h"
#include "graphics/vertexArray.h"
#include "graphics/window.h"
#include "ImGuiLayer.h"
#include "geometry/mesh.h"
#include "io/reader.h"
#include "panels/settingsPanel.h"
//#include "panels/hierarchyPanel.h"

#include "FL/Fl_Simple_Terminal.H"
#include "glm/fwd.hpp"
#include <GLFW/glfw3.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace FLB 
{
  // forward declaration to prevent including entt library in cuda files (produce errors in the compilation)
  class Scene;
  class HierarchyPanel;
  class MetricsPanel;
  class IsosurfacePanel;
  class ConsultValuesPanel;

  class RenderLayer
  {
    public:
      RenderLayer() = default;
      RenderLayer(FLB::Mesh* mesh, const FLB::OptionsCalculation& optionsCalc, Fl_Simple_Terminal* terminal);
      ~RenderLayer();

      void onUpdate();

      void onAttach(FLB::Mesh* mesh, const FLB::OptionsCalculation& optionsCalc, Fl_Simple_Terminal* terminal);

      void onImGuiRender();

      FLB::Window* getRenderWindow() const {return m_RenderWindow.get();}
      FLB::FrameBuffer* getFramebuffer() const {return m_Framebuffer.get();}
      FLB::Scene* getScene() const {return m_Scene.get();}

      const int* getTypeRendering() const {return m_TypeRendering;}
      const bool* isIsosurfaceRendering() const;
      const FLB::IsoSurfaceComponent& getIsoSurfaceComponent() const;
      void getCameraDomainBounds(unsigned int* cameraBounds) const {m_OrthographicCameraController -> getCameraDomainBounds(cameraBounds);}
      float getSecondsFrameRate() const {return m_SecondsFrameRate;}

      void setTime(float time) {m_Time = time;}
      void setCalculationTimeStats(float time) {m_MiliSecondsSimulation = time * 1000.0f; m_FrameRateSimulation = 1.0f/time;}
      void setUsedFreeMemoryGPU(std::array<float, 2> usedFreeMemory);

    private:

      void addGlyph();
      void addIsosurface();
      void addScalarVectorField();

      const FLB::Mesh* m_DomainMesh;

      const int* m_TypeRendering; // point or texture

      FLB::API m_Api;
      std::unique_ptr<FLB::ImGuiLayer> m_ImGuiLayer;

      std::unique_ptr<FLB::OrthographicCameraController> m_OrthographicCameraController;
      std::unique_ptr<FLB::FrameBuffer> m_Framebuffer;
      std::unique_ptr<FLB::Window> m_RenderWindow;
      GLFWwindow* m_GLFWwindow;
      Fl_Simple_Terminal* m_Terminal;

      std::unique_ptr<FLB::HierarchyPanel> m_HierarchyPanel;
      std::unique_ptr<FLB::MetricsPanel> m_MetricsPanel;
      std::unique_ptr<FLB::SettingsPanel2D> m_SettingsPanel2D;
      std::unique_ptr<FLB::IsosurfacePanel> m_IsosurfacePanel;
      std::unique_ptr<FLB::ConsultValuesPanel> m_ConsultValuesPanel;

      std::unique_ptr<FLB::Scene> m_Scene;

      glm::uvec2 m_ViewportSize = {0, 0};
      glm::vec2 m_ViewportBounds[2]; // min and max of viewport
      
      bool m_ViewportFocused = false;
      bool m_ViewportHovered = false;
      bool m_ShowHierarchyPanel = true;
      bool m_ShowMetricsPanel = false;
      bool m_ShowIsosurfacePanel = false;
      bool m_ShowSettingsPanel = false;
      bool m_ShowConsultValuesPanel = false;
      bool m_Is3D = false;

      int m_GizmoOperation = -1; // Translate = 0, Rotate = 1, Scale = 2, Bounds = 3
      
      // time is SI units
      float m_Time = 0.0f;

      float m_MiliSecondsSimulation = 0.0f;
      float m_FrameRateSimulation = 0.0f;

      // framerate limit for the rendering of the calculation
      int m_FramerateLimit = 30;
      float m_SecondsFrameRate = 1.0f/30.0f;

      friend class SettingsPanel2D;
  };

}
