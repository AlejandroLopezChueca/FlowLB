#pragma once

#include "graphics/API.h"
#include "graphics/cameraController.h"
#include "graphics/frameBuffer.h"
#include "graphics/vertexArray.h"
#include "graphics/window.h"
#include "ImGuiLayer.h"
#include "geometry/mesh.h"
#include "io/reader.h"
//#include "panels/hierarchyPanel.h"

#include "FL/Fl_Simple_Terminal.H"
#include "glm/fwd.hpp"
#include <GLFW/glfw3.h>
#include <cstddef>
#include <memory>

namespace FLB 
{
  // forward declaration to prevent including entt library in cuda files (produce errors in the compilation)
  class Scene;
  class HierarchyPanel;
  class MetricsPanel;

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

      void setTime(float time) {m_Time = time;}

    private:

      void addGlyph();
      void addScalarVectorField();

      const FLB::Mesh* m_DomainMesh;

      const int* m_TypeRendering;

      FLB::API m_Api;
      std::unique_ptr<FLB::ImGuiLayer> m_ImGuiLayer;

      std::unique_ptr<FLB::OrthographicCameraController> m_OrthographicCameraController;
      std::unique_ptr<FLB::FrameBuffer> m_Framebuffer;
      std::unique_ptr<FLB::Window> m_RenderWindow;
      GLFWwindow* m_GLFWwindow;
      Fl_Simple_Terminal* m_Terminal;

      std::unique_ptr<FLB::HierarchyPanel> m_HierarchyPanel;
      std::unique_ptr<FLB::MetricsPanel> m_MetricsPanel;

      std::unique_ptr<FLB::Scene> m_Scene;

      glm::uvec2 m_ViewportSize = {0, 0};
      glm::vec2 m_ViewportBounds[2];

      bool m_ViewportFocused = false;
      bool m_ViewportHovered = false;
      bool m_ShowHierarchyPanel = true;
      bool m_ShowMetricsPanel = false;
      bool m_Is3D = false;

      int m_GizmoOperation = -1; // Translate = 0, Rotate = 1, Scale = 2, Bounds = 3
      
      // time is SI units
      float m_Time = 0.0f;
  };

}
