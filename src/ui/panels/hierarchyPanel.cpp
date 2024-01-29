#include "hierarchyPanel.h"
#include "glm/gtc/type_ptr.hpp"
#include "glm/trigonometric.hpp"
#include "graphics/scene/components.h"
#include "graphics/texture.h"
#include "ui/uiUtils.h"
#include "graphics/colorMaps.h"
#include "graphics/renderer.h"

#include <imgui.h>
#include <cstdint>
#include <memory>
#include <string>
#include <iostream>

std::array<std::unique_ptr<FLB::Texture2D>, 4> FLB::HierarchyPanel::s_ColorMaps;

FLB::HierarchyPanel::HierarchyPanel(FLB::API api)
{
  createTextures(api);
}

void FLB::HierarchyPanel::setScene(FLB::Scene* scene)
{
  m_Scene = scene;
  m_SelectedEntity = {};
}

void FLB::HierarchyPanel::createTextures(FLB::API api)
{
  s_ColorMaps[0] = FLB::Texture2D::create(api, "resources/icons/colorMaps/gray.png");
  s_ColorMaps[1] = FLB::Texture2D::create(api, "resources/icons/colorMaps/imola.png");
  s_ColorMaps[2] = FLB::Texture2D::create(api, "resources/icons/colorMaps/roma.png");
  s_ColorMaps[3] = FLB::Texture2D::create(api, "resources/icons/colorMaps/devon.png");
}

void FLB::HierarchyPanel::onImGuiRender(bool* open)
{
  ImGui::SetNextWindowSize(ImVec2(200,500), ImGuiCond_FirstUseEver);
  ImGui::Begin("Pipeline", open);
  drawScalarVectorialFieldNode();

  for (auto entityID : m_Scene -> m_Registry.storage<entt::entity>())
  {
    FLB::Entity entity{entityID, m_Scene};
    drawEntityNode(entity);
  }

  if (ImGui::IsMouseDown(0) && ImGui::IsWindowHovered())
  {
    m_SelectedEntity = {};
    m_ScalarVectorialFieldSelected = false;
    m_Scene -> m_SelectedEntity = &m_SelectedEntity;
  }

  ImGui::End();

  ImGui::Begin("Properties", open);
  if (m_ScalarVectorialFieldSelected) drawScalarVectorialFieldProperties();
  else if (m_SelectedEntity) drawComponents();
  ImGui::End();
}

void FLB::HierarchyPanel::drawEntityNode(FLB::Entity& entity)
{
  const std::string& tag = entity.getName();
  bool& draw = entity.getDrawAction();
  ImGui::SetNextItemAllowOverlap();
  ImGui::PushID((uint32_t)entity);
  ImGui::Checkbox("##", &draw);
  ImGui::PopID();
  ImGui::SameLine();
  
  ImGuiDockNodeFlags flags = ((m_SelectedEntity == entity) ? ImGuiTreeNodeFlags_Selected : 0) | ImGuiTreeNodeFlags_Bullet | ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_NoTreePushOnOpen;
  ImGui::TreeNodeEx(tag.c_str(), flags);
  
  if (ImGui::IsItemClicked()) m_SelectedEntity = entity;

  if (ImGui::BeginPopupContextItem())
  {
    if (ImGui::MenuItem("Delete Entity"))
    {
      if (m_SelectedEntity == entity)
      {
	m_Scene -> destroyEntity(entity);
	m_SelectedEntity = {};
	m_Scene -> m_SelectedEntity = &m_SelectedEntity;
      }
    }
    ImGui::EndPopup();
  }
}

void FLB::HierarchyPanel::drawScalarVectorialFieldNode()
{
  static const std::string tag = "Scalar/Vectorial Field";
  ImGuiDockNodeFlags flags = (m_ScalarVectorialFieldSelected ? ImGuiTreeNodeFlags_Selected : 0) | ImGuiTreeNodeFlags_Bullet | ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_NoTreePushOnOpen;
  ImGui::TreeNodeEx(tag.c_str(), flags);

  if (ImGui::IsItemClicked()) m_ScalarVectorialFieldSelected = true;
}

void FLB::HierarchyPanel::drawComponents()
{
  FLB::UiUtils::drawComponent<FLB::Arrow2DComponent>("Glyph Definition", m_SelectedEntity, [](auto& component)
  {
    glm::ivec2 prevNumberPoints = component.numberPoints;
    FLB::UiUtils::drawIntVec2Control("Number of points", component.numberPoints);
    if (prevNumberPoints.x != component.numberPoints.x || prevNumberPoints.y != component.numberPoints.y) component.updateMesh = true;
    else component.updateMesh = false;
    
    ImGui::SliderFloat("Scale", &component.scale, 0.0f, 1.0f, "scale = %.3f");
    
    ImGui::SeparatorText("Gradient");
    ImGui::DragFloatRange2("##", &component.maxMinValues.y, &component.maxMinValues.x, 0.05f, 0.0f, 0.0f, "Min: %.2f", "Max: %.2f");
    FLB::UiUtils::drawImageComboBox<4>(s_ColorMaps, FLB::ColorMaps::nameColorMaps, component.currentIdxColorMap);
  });
  
  FLB::UiUtils::drawComponent<FLB::RectangleComponent>("Rectangle Definition", m_SelectedEntity, m_Scene, [](FLB::Entity& entity, auto& component, auto* scene)
  {
    scene -> m_SelectedEntity = &entity;

    bool prevDraw = component.draw;
    ImGui::Checkbox("Draw", &component.draw);

    if (component.draw > prevDraw) scene -> m_NumActiveRectangles += 1;
    else if (component.draw < prevDraw) scene -> m_NumActiveRectangles -= 1;

    FLB::UiUtils::drawVec2Control("Dimensions", component.dimensions);
    ImGui::ColorEdit4("Color", glm::value_ptr(component.color));
  });
  
  FLB::UiUtils::drawComponent<FLB::TransformComponent>("Rectangle Transform", m_SelectedEntity, [](auto& component)
  {
    FLB::UiUtils::drawVec3Control("Translation", component.translation);
    
    glm::vec3 rotation = glm::degrees(component.rotation);
    FLB::UiUtils::drawVec3Control("Rotation", rotation);
    component.rotation = glm::radians(rotation);
    
    FLB::UiUtils::drawVec3Control("Scale", component.scale, {1.0f, 1.0f, 1.0f});
  });
}

void FLB::HierarchyPanel::drawScalarVectorialFieldProperties()
{
  auto& scalarVectorialFieldComponent = m_Scene -> m_ScalarVectorialFieldComponent;
  ImGuiTreeNodeFlags treeNodeFlags = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_AllowItemOverlap | ImGuiTreeNodeFlags_FramePadding;
  if (ImGui::TreeNodeEx("Representation", treeNodeFlags, "Representation"))
  {
    int prevIdxField = 10 * scalarVectorialFieldComponent.type;
    FLB::UiUtils::drawComboBox(m_AvailableFields, scalarVectorialFieldComponent.idField, "Field");
    ImGui::RadioButton("Point Data", &scalarVectorialFieldComponent.type, 0);
    ImGui::SameLine();
    ImGui::RadioButton("Texture Data", &scalarVectorialFieldComponent.type, 1);
    switch (scalarVectorialFieldComponent.idField) 
    { 
      case 0: // Velocity
      {
	prevIdxField += m_CurrentIdxVectorRepresentation;
	FLB::UiUtils::drawComboBox(m_NameVectorRepresentation, m_CurrentIdxVectorRepresentation, "Value");
	int newIdxField = 10 * scalarVectorialFieldComponent.type + m_CurrentIdxVectorRepresentation; 
	if (prevIdxField != newIdxField) FLB::Renderer::setShaderInUse(newIdxField);
	break;
      }
      case 1: // Pressure
      {
	break;
      } 
    }

    ImGui::TreePop();
  }

  if (ImGui::TreeNodeEx("Coloring", treeNodeFlags, "Coloring"))
  {
    ImGui::SeparatorText("Gradient");
    ImGui::DragFloatRange2("##", &scalarVectorialFieldComponent.maxMinValues.y, &scalarVectorialFieldComponent.maxMinValues.x, 0.05f, 0.0f, 0.0f, "Min: %.2f", "Max: %.2f");
    FLB::UiUtils::drawImageComboBox<4>(s_ColorMaps, FLB::ColorMaps::nameColorMaps, scalarVectorialFieldComponent.currentIdxColorMap);
    ImGui::TreePop();
  }

  if (ImGui::TreeNodeEx("Styling", treeNodeFlags, "Styling"))
  {
    if (scalarVectorialFieldComponent.type == 0)
    {
      ImGui::InputFloat("Size points", &scalarVectorialFieldComponent.sizePoints, 0.01f, 1.0f, "%.3f");
    }
    ImGui::TreePop();
  }

}
