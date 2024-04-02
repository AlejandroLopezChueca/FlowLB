#include "renderLayer.h"
#include "graphics/cameraController.h"
#include "panels/hierarchyPanel.h"
#include "panels/addPanel.h"
#include "ImGuiLayer.h"
#include "app.h"
#include "geometry/mesh.h"
#include "graphics/API.h"
#include "graphics/renderer.h"
#include "graphics/vertexArray.h"
#include "graphics/frameBuffer.h"
#include "graphics/window.h"
#include "io/reader.h"
#include "graphics/scene/scene.h"
#include "graphics/scene/entity.h"
#include "graphics/scene/components.h"
#include "math/math.h"


#include "imgui.h"
#include "imgui_internal.h"
#include "ImGuizmo.h"
#include "panels/settingsPanel.h"
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <glm/gtc/type_ptr.hpp>


FLB::RenderLayer::RenderLayer(FLB::Mesh* mesh, const FLB::OptionsCalculation& optionsCalc, Fl_Simple_Terminal* terminal)
  :m_Api(optionsCalc.graphicsAPI), m_Terminal(terminal)
{
  onAttach(mesh, optionsCalc, terminal);
}

FLB::RenderLayer::~RenderLayer()
{
  FLB::Renderer::resetData();
}

void FLB::RenderLayer::onUpdate()
{
  // resize if necessary
  FrameBufferSpecifications& specs = m_Framebuffer -> getSpecifications();

  if (m_ViewportSize.x > 0 && m_ViewportSize.y > 0 && (specs.width != m_ViewportSize.x || specs.height != m_ViewportSize.y))
  {
    m_Framebuffer -> resize(m_ViewportSize.x, m_ViewportSize.y);
    m_OrthographicCameraController -> setViewportSize((float)m_ViewportSize.x, (float)m_ViewportSize.y);
  }

  // Render
  m_Framebuffer -> bind();
  FLB::Renderer::setClearColor({0.5f, 0.5f, 0.5f, 1});
  FLB::Renderer::clear();
  FLB::Renderer::beginScene(m_OrthographicCameraController -> getCamera());

  m_Scene -> update(m_OrthographicCameraController.get());
  // update camera after scene to synchronize the creation of isosurfaces in cuda with the visualization with OpenGL
  m_OrthographicCameraController -> GLFWUpdate(m_GLFWwindow);
  
  m_Framebuffer -> unbind();

  m_ImGuiLayer -> begin();
  onImGuiRender();
  m_ImGuiLayer -> onImGuiRender();
  m_ImGuiLayer -> end();
  m_RenderWindow -> update();
}

void FLB::RenderLayer::onAttach(FLB::Mesh* mesh, const FLB::OptionsCalculation& optionsCalc, Fl_Simple_Terminal* terminal)
{
  m_DomainMesh = mesh;
  m_Is3D = optionsCalc.typeAnalysis;

  // Init Renderer
  // Set the API to use for the render
  FLB::RendererAPI::setAPI(optionsCalc.graphicsAPI);
  // Set the renderer to call the API
  FLB::Renderer::setRendererAPI(mesh -> getIndicesCorners());
  
  // Options used in 2D or 3D
  if (optionsCalc.typeAnalysis == 0) //2D
  {
    m_OrthographicCameraController = std::make_unique<FLB::OrthographicCameraController>(1600.0f, 900.0f);
    m_OrthographicCameraController -> setDomainData(mesh);
    m_RenderWindow = FLB::Window::create<FLB::OrthographicCameraController>(optionsCalc.graphicsAPI, m_OrthographicCameraController.get(), optionsCalc.typeAnalysis, terminal);
    m_RenderWindow -> setGizmoOperation(&m_GizmoOperation);
    ImGuizmo::SetOrthographic(true);
  
    m_SettingsPanel2D = std::make_unique<FLB::SettingsPanel2D>(m_OrthographicCameraController.get(), this);
  }
  else if (optionsCalc.typeAnalysis == 1) //3D
  {

  }
  // Framebuffer 
  FLB::FrameBufferSpecifications specs;
  specs.width = 1600;
  specs.height = 900;
  m_Framebuffer = FLB::FrameBuffer::create(optionsCalc.graphicsAPI, specs);
  // Before the init is necesary to create the context (generate witt the creation of the window)
  FLB::Renderer::init(optionsCalc, terminal, mesh);
  
  // After creating the context, it is necesarry to setup the mesh
  mesh -> setupGraphicsOptions(optionsCalc.graphicsAPI, terminal);
  // Add quad vertexbuffer to mesh vertex array for 2D analysis
  if (optionsCalc.typeAnalysis == 0) mesh -> addVertexBuffer(FLB::Renderer::getVertexBufferQuad());

  m_GLFWwindow = static_cast<GLFWwindow*>(m_RenderWindow -> getWindow());

  m_ImGuiLayer = std::make_unique<FLB::ImGuiLayer>();
  m_ImGuiLayer -> setContext(m_RenderWindow.get());

  m_Scene = std::make_unique<FLB::Scene>(mesh);
  if (optionsCalc.typeAnalysis == 0) m_Scene -> setOrthographicCameraController(m_OrthographicCameraController.get());
  m_Scene -> setCalculationAPI(optionsCalc.calculationAPI);

  m_TypeRendering = m_Scene -> getRenderingScalarVectorialField();

  m_HierarchyPanel = std::make_unique<FLB::HierarchyPanel>(optionsCalc.graphicsAPI);
  m_HierarchyPanel -> setScene(m_Scene.get());

  m_MetricsPanel = std::make_unique<FLB::MetricsPanel>(optionsCalc.graphicsAPI, mesh, m_Time, m_MiliSecondsSimulation, m_FrameRateSimulation);
  
  m_IsosurfacePanel = std::make_unique<FLB::IsosurfacePanel>(optionsCalc.graphicsAPI, optionsCalc.calculationAPI, terminal, mesh);
  m_IsosurfacePanel ->setScene(m_Scene.get());

  if (optionsCalc.typeAnalysis == 0) m_ConsultValuesPanel = std::make_unique<FLB::ConsultValuesPanel>(FLB::Renderer::getScalarFieldsTexture(), m_RenderWindow.get(), m_OrthographicCameraController.get(), m_HierarchyPanel -> getCurrentIdxVectorRepresentation(), m_Scene -> getScalarVectorialFieldComponent().idField);
}

void FLB::RenderLayer::onImGuiRender()
{
  // Note: Switch to true to enable dockspace
  static bool dockspaceOpen = true;
  static bool opt_fullscreen_persistant = true;
  bool opt_fullscreen = opt_fullscreen_persistant;
  static ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_None;

  // We are using the ImGuiWindowFlags_NoDocking flag to make the parent window not dockable into,
  // because it would be confusing to have two docking targets within each others.
  ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
  if (opt_fullscreen)
  {
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(viewport->Size);
    ImGui::SetNextWindowViewport(viewport->ID);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
    window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
  }

  // When using ImGuiDockNodeFlags_PassthruCentralNode, DockSpace() will render our background and handle the pass-thru hole, so we ask Begin() to not render a background.
  if (dockspace_flags & ImGuiDockNodeFlags_PassthruCentralNode) window_flags |= ImGuiWindowFlags_NoBackground;

  // Important: note that we proceed even if Begin() returns false (aka window is collapsed).
  // This is because we want to keep our DockSpace() active. If a DockSpace() is inactive, 
  // all active windows docked into it will lose their parent and become undocked.
  // We cannot preserve the docking relationship between an active window and an inactive docking, otherwise 
  // any change of dockspace/settings would lead to windows being stuck in limbo and never being visible.
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
  ImGui::Begin("DockSpaceDemo", &dockspaceOpen, window_flags);
  ImGui::PopStyleVar(); // end 0 padding for viewport
			//
  ImGui::SetKeyOwner(ImGuiMod_Alt,1); // disable Alt key

  if (opt_fullscreen) ImGui::PopStyleVar(2);

  // Submit the DockSpace
  ImGuiIO& io = ImGui::GetIO();
  ImGuiStyle& style = ImGui::GetStyle();
  float minWinSizeX = style.WindowMinSize.x;
  style.WindowMinSize.x = 370.0f;
  if (io.ConfigFlags & ImGuiConfigFlags_DockingEnable)
  {
    ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
    ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);
  }

  style.WindowMinSize.x = minWinSizeX;
  // Check visualization panels of the menuBar
  if(ImGui::BeginMenuBar())
  {
    if (ImGui::BeginMenu("File"))
    {
      if (ImGui::MenuItem("Exit")) FLB::App::closeGraphics();
      ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("View"))
    {
      if (ImGui::MenuItem("Hierarchy panel")) m_ShowHierarchyPanel = true;
      ImGui::Separator();
      if (ImGui::MenuItem("Metrics")) m_ShowMetricsPanel = true;
      if (ImGui::MenuItem("Consult Values")) m_ShowConsultValuesPanel = true;
      ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Settings"))
    {
      if (ImGui::MenuItem("General Settings")) m_ShowSettingsPanel = true; 
      ImGui::EndMenu();
    }
    
    if (ImGui::BeginMenu("Add"))
    {
      if (ImGui::MenuItem("Scalar/Vectorial Field")) addScalarVectorField();
      if (ImGui::MenuItem("Glyphs")) addGlyph();
      if (ImGui::MenuItem("Isosurface")) 
      {
	FLB::Renderer::s_UpdateRender = false;
	m_ShowIsosurfacePanel = true;
      }
      ImGui::EndMenu();
    }
    ImGui::EndMenuBar();
  }

  // show panels
  if (m_ShowHierarchyPanel) m_HierarchyPanel -> onImGuiRender(&m_ShowHierarchyPanel);
  if (m_ShowMetricsPanel) m_MetricsPanel -> onImGuiRender(&m_ShowMetricsPanel);
  if (m_ShowSettingsPanel) m_SettingsPanel2D -> onImGuiRender(&m_ShowSettingsPanel);
  if (m_ShowIsosurfacePanel) m_IsosurfacePanel -> onImGuiRender(&m_ShowIsosurfacePanel);
  if (m_ShowConsultValuesPanel) m_ConsultValuesPanel -> onImGuiRender(&m_ShowConsultValuesPanel, m_ViewportBounds);

  // Viewport
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{ 0, 0 });
  ImGui::Begin("Viewport");

  ImVec2 viewportMinRegion = ImGui::GetWindowContentRegionMin();
  ImVec2 viewportMaxRegion = ImGui::GetWindowContentRegionMax();
  ImVec2 viewportOffset = ImGui::GetWindowPos();
  m_ViewportBounds[0] = { viewportMinRegion.x + viewportOffset.x, viewportMinRegion.y + viewportOffset.y };
  m_ViewportBounds[1] = { viewportMaxRegion.x + viewportOffset.x, viewportMaxRegion.y + viewportOffset.y };

  m_ViewportFocused = ImGui::IsWindowFocused();

  ImVec2 viewportPanelSize = ImGui::GetContentRegionAvail();
  m_ViewportSize = {viewportPanelSize.x, viewportPanelSize.y};

  uint64_t textureID = m_Framebuffer -> getTextureColorID();
  ImGui::Image((void*)textureID, viewportPanelSize, ImVec2{0,1}, ImVec2{1,0});

  // Gizmos
  const FLB::Entity& selectedEntity = m_HierarchyPanel -> getSelectedEntity();
  if (selectedEntity && m_GizmoOperation != -1)
  {
    ImGuizmo::SetDrawlist();
    ImGui::GetWindowWidth();
    ImGuizmo::SetRect(m_ViewportBounds[0].x, m_ViewportBounds[0].y, m_ViewportBounds[1].x - m_ViewportBounds[0].x, m_ViewportBounds[1].y - m_ViewportBounds[0].y);

    // camera
    const glm::mat4& cameraProjection = m_OrthographicCameraController -> getCamera().getProjectionMatrix();
    const glm::mat4& cameraView = m_OrthographicCameraController -> getCamera().getViewMatrix();

    // Entity transform
    auto& transformComponent = selectedEntity.getComponent<FLB::TransformComponent>();
    glm::mat4 transform = transformComponent.getTransform();

    ImGuizmo::Manipulate(glm::value_ptr(cameraView), glm::value_ptr(cameraProjection), (ImGuizmo::OPERATION)m_GizmoOperation, ImGuizmo::LOCAL, glm::value_ptr(transform));

    if (ImGuizmo::IsUsing())
    {
      glm::vec3 translation, rotation, scale;
      FLB::Math::decomposeTransform(transform, translation, rotation, scale);
      glm::vec3 deltaRotation = rotation - transformComponent.rotation;
      transformComponent.translation = translation;
      transformComponent.rotation += deltaRotation;
      transformComponent.scale = scale; 
    }
  }

  ImGui::End(); // end viewport
  ImGui::PopStyleVar(); 

  // Principal panels
  //UIToolBar();
  //m_ContentBrowserPanel.onImGuiRender();

  ImGui::End(); // end dockspace
}

const bool* FLB::RenderLayer::isIsosurfaceRendering() const
{
  return m_Scene -> isIsosurfaceRendering();
}

const FLB::IsoSurfaceComponent& FLB::RenderLayer::getIsoSurfaceComponent() const
{
  auto view = m_Scene -> getRegistry().view<FLB::IsoSurfaceComponent>();
  // Return only the first
  return view.get<FLB::IsoSurfaceComponent>(*view.begin());
}

void FLB::RenderLayer::setUsedFreeMemoryGPU(std::array<float, 2> usedFreeMemory)
{
  m_MetricsPanel ->m_UsedMemory = usedFreeMemory[0];
  m_MetricsPanel ->m_FreeMemory = usedFreeMemory[1];
}

void FLB::RenderLayer::addGlyph()
{
  // Stop calculations to facilitate the characterization of the entity
  FLB::Renderer::s_UpdateRender = false;
  FLB::Entity glyphEntity = m_Scene -> createEntity("Glyph");
  if (m_Is3D) 
  {

  }
  else 
  {
    auto& drawComponent = glyphEntity.getComponent<FLB::DrawComponent>();


    glyphEntity.addComponent<FLB::Arrow2DComponent>(m_Api, m_Terminal, &drawComponent.draw, m_DomainMesh);
    glyphEntity.addComponent<FLB::RectangleComponent>();
    glyphEntity.addComponent<FLB::TransformComponent>();
    FLB::Renderer::addInstanceRectangles(m_Api, m_Terminal, m_DomainMesh);

    uint32_t rectangleInstanceCount = FLB::Renderer::getInstanceCountRectangles();
    auto& rectangleComponent = glyphEntity.getComponent<FLB::RectangleComponent>();
    rectangleComponent.idx = rectangleInstanceCount - 1;

    // correct transform to translate the rectangle to the center of the domain and scale to the domain (only if the longitudes are less than 1)
    const glm::vec2& centerCoordinates = FLB::Renderer::getRectangleMesh() ->getCenterCoordinates();
    auto& transformComponent = glyphEntity.getComponent<FLB::TransformComponent>();
    transformComponent.translation = {centerCoordinates.x, centerCoordinates.y, 0.0f};

    float xLongitudeDomain = m_DomainMesh -> getxMax() - m_DomainMesh -> getxMin();
    xLongitudeDomain = xLongitudeDomain > 1.0f ? 1.0f : xLongitudeDomain;
    float yLongitudeDomain = m_DomainMesh -> getyMax() - m_DomainMesh -> getyMin();
    yLongitudeDomain = yLongitudeDomain > 1.0f ? 1.0f : yLongitudeDomain;
    transformComponent.scale = {xLongitudeDomain, yLongitudeDomain, 1.0f};
  }
}

void FLB::RenderLayer::addIsosurface()
{
  FLB::Renderer::s_UpdateRender = false;
}

void FLB::RenderLayer::addScalarVectorField()
{
  // Stop calculations to facilitate the characterization of the entity
  FLB::Renderer::s_UpdateRender = false;
}
