#pragma once

#include "graphics/texture.h"
#include "graphics/scene/entity.h"

#include <vector>
#include <string>
#include "imgui.h"
#include "imgui_internal.h"


namespace  FLB::UiUtils
{
  void drawIntVec2Control(const std::string& label, glm::ivec2& values, const glm::ivec2& resetValues = {2, 2}, float columnWidth = 150.0f);
  
  void drawVec2Control(const std::string& label, glm::vec2& values, const glm::vec2& resetValues = {0.0f, 0.0f}, float columnWidth = 150.0f);

  void drawVec3Control(const std::string& label, glm::vec3& values, const glm::vec3& resetValues = {0.0f, 0.0f, 0.0f}, float columnWidth = 100.0f);

  template<typename T, typename UIFunction>
  void drawComponent(const std::string& name, FLB::Entity& entity, UIFunction uiFunction)
  {	
    const ImGuiTreeNodeFlags treeNodeFlags = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_AllowItemOverlap | ImGuiTreeNodeFlags_FramePadding;
    if (entity.hasComponent<T>())
    {
      auto& component = entity.getComponent<T>();
      ImVec2 contentRegionAvailable = ImGui::GetContentRegionAvail();

      ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2{ 4, 4 });
      float lineHeight = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
      ImGui::Separator();
      bool open = ImGui::TreeNodeEx((void*)typeid(T).hash_code(), treeNodeFlags, name.c_str());
      ImGui::PopStyleVar();

      ImGui::SameLine(contentRegionAvailable.x - lineHeight * 0.5f);
      if (ImGui::Button("+", ImVec2{ lineHeight, lineHeight })) ImGui::OpenPopup("ComponentSettings");
      bool removeComponent = false;
      if (ImGui::BeginPopup("ComponentSettings"))
      {
	if (ImGui::MenuItem("Remove component")) removeComponent = true;
	ImGui::EndPopup();
      }

      if (open)
      {
	uiFunction(component);
	ImGui::TreePop();
      }

      if (removeComponent) entity.removeComponent<T>();
    }
  }
  
  template<typename T, typename UIFunction>
  void drawComponent(const std::string& name, FLB::Entity& entity, FLB::Scene* scene, UIFunction uiFunction)
  {	
    const ImGuiTreeNodeFlags treeNodeFlags = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_AllowItemOverlap | ImGuiTreeNodeFlags_FramePadding;
    if (entity.hasComponent<T>())
    {
      auto& component = entity.getComponent<T>();
      ImVec2 contentRegionAvailable = ImGui::GetContentRegionAvail();

      ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2{ 4, 4 });
      float lineHeight = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
      ImGui::Separator();
      bool open = ImGui::TreeNodeEx((void*)typeid(T).hash_code(), treeNodeFlags, name.c_str());
      ImGui::PopStyleVar();

      ImGui::SameLine(contentRegionAvailable.x - lineHeight * 0.5f);
      if (ImGui::Button("+", ImVec2{ lineHeight, lineHeight })) ImGui::OpenPopup("ComponentSettings");
      bool removeComponent = false;
      if (ImGui::BeginPopup("ComponentSettings"))
      {
	if (ImGui::MenuItem("Remove component")) removeComponent = true;
	ImGui::EndPopup();
      }

      if (open)
      {
	uiFunction(entity, component, scene);
	ImGui::TreePop();
      }

      if (removeComponent) entity.removeComponent<T>();
    }
  }

  void drawComboBox(std::vector<std::string>& datasetName, int& currentIdx, const char* name);
    
  void helpMarker(const char* description);

  template<int T>
  void drawImageComboBox(std::array<std::unique_ptr<FLB::Texture2D>, T>& images, std::array<std::string, T>& namesImages, int& currentIdx)
  {
    ImVec2 comboPos = ImGui::GetCursorScreenPos();
    if(ImGui::BeginCombo("Colormap", ""))
    {
      for (int i = 0; i < images.size(); i++)
      {
	ImGui::GetWindowPos();
	bool isSelected = (currentIdx == i);
	ImGui::PushID(i);
	if (ImGui::Selectable("", isSelected)) currentIdx = i;
	ImGui::SameLine(0,0);
	uint32_t textureID = images[i] -> getID();
	float height = ImGui::GetTextLineHeight();
	ImGui::Image(reinterpret_cast<ImTextureID>(textureID),ImVec2{10 * height, height}, ImVec2{0,0}, ImVec2{1,1});
	ImGui::SameLine();
	ImGui::Text(namesImages[i].c_str());
	if (isSelected) ImGui::SetItemDefaultFocus();
	ImGui::PopID();
      }
      ImGui::EndCombo();
    }
    // Reflect images selected in the ComboBox
    ImVec2 backupPos = ImGui::GetCursorScreenPos();
    ImGuiStyle& style = ImGui::GetStyle();
    ImGui::SetCursorScreenPos(ImVec2{comboPos.x + style.FramePadding.x, comboPos.y});
    float height = ImGui::GetTextLineHeight();
    uint32_t textureID = images[currentIdx] -> getID();
    ImGui::Image(reinterpret_cast<ImTextureID>(textureID),ImVec2{10 * height, height}, ImVec2{0,0}, ImVec2{1,1});
    ImGui::SameLine();
    ImGui::Text(namesImages[currentIdx].c_str());
    
    ImGui::SetCursorScreenPos(backupPos);
  }
}
