#include "uiUtils.h"
#include "imgui.h"
  
void FLB::UiUtils::drawIntVec2Control(const std::string& label, glm::ivec2& values, const glm::ivec2& resetValues, float columnWidth)
{
  ImGuiIO& io = ImGui::GetIO();
  auto boldFont = io.Fonts -> Fonts[0];

  ImGui::PushID(label.c_str());

  ImGui::BeginTable("##", 2);
  ImGui::TableSetupColumn("##1", ImGuiTableColumnFlags_WidthFixed, columnWidth);
  ImGui::TableSetupColumn("##2", ImGuiTableColumnFlags_WidthFixed,  2 * columnWidth);

  ImGui::TableNextRow();
  ImGui::TableSetColumnIndex(0);
  ImGui::Text(label.c_str());
  ImGui::TableNextColumn();
  
  ImGui::PushMultiItemsWidths(2, ImGui::CalcItemWidth());
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0.0f, 0.0f});
  
  float lineHeight = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
  ImVec2 buttonSize = { lineHeight + 3.0f, lineHeight };
  
  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.8f, 0.1f, 0.15f, 1.0f });
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.9f, 0.2f, 0.2f, 1.0f });
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.8f, 0.1f, 0.15f, 1.0f });
  ImGui::PushFont(boldFont);
  if (ImGui::Button("X")) values.x = resetValues.x;
  ImGui::PopFont();
  ImGui::PopStyleColor(3);

  ImGui::SameLine();
  ImGui::DragInt("##X", &values.x, 1, 2, INT_MAX, "%d", ImGuiSliderFlags_AlwaysClamp);

  ImGui::PopItemWidth();
  ImGui::SameLine();

  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f });
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.3f, 0.8f, 0.3f, 1.0f });
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f });  
  ImGui::PushFont(boldFont);
  if (ImGui::Button("Y")) values.y = resetValues.y;
  ImGui::PopFont();
  ImGui::PopStyleColor(3);

  ImGui::SameLine();
  ImGui::DragInt("##Y", &values.y, 1, 2, INT_MAX, "%d", ImGuiSliderFlags_AlwaysClamp);
  ImGui::PopItemWidth();

  ImGui::PopStyleVar();

  ImGui::EndTable();

  ImGui::PopID();

}

void FLB::UiUtils::drawVec2Control(const std::string& label, glm::vec2& values, const glm::vec2& resetValues, float columnWidth)
{
  ImGuiIO& io = ImGui::GetIO();
  auto boldFont = io.Fonts -> Fonts[0];

  ImGui::PushID(label.c_str());

  ImGui::BeginTable("##", 2);
  ImGui::TableSetupColumn("##1", ImGuiTableColumnFlags_WidthFixed, columnWidth);
  ImGui::TableSetupColumn("##2", ImGuiTableColumnFlags_WidthFixed,  2 * columnWidth);

  ImGui::TableNextRow();
  ImGui::TableSetColumnIndex(0);
  ImGui::Text(label.c_str());
  ImGui::TableNextColumn();
  
  ImGui::PushMultiItemsWidths(2, ImGui::CalcItemWidth());
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0.0f, 0.0f});
  
  float lineHeight = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
  ImVec2 buttonSize = { lineHeight + 3.0f, lineHeight };
  
  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.8f, 0.1f, 0.15f, 1.0f });
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.9f, 0.2f, 0.2f, 1.0f });
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.8f, 0.1f, 0.15f, 1.0f });
  ImGui::PushFont(boldFont);
  if (ImGui::Button("X")) values.x = resetValues.x;
  ImGui::PopFont();
  ImGui::PopStyleColor(3);

  ImGui::SameLine();
  ImGui::DragFloat("##X", &values.x, 0.05f, 0.0f, FLT_MAX, "%.2f");
  ImGui::PopItemWidth();
  ImGui::SameLine();

  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f });
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.3f, 0.8f, 0.3f, 1.0f });
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f });  
  ImGui::PushFont(boldFont);
  if (ImGui::Button("Y")) values.y = resetValues.y;
  ImGui::PopFont();
  ImGui::PopStyleColor(3);

  ImGui::SameLine();
  ImGui::DragFloat("##Y", &values.y, 0.05f, 0.0f, FLT_MAX, "%.2f");
  ImGui::PopItemWidth();

  ImGui::PopStyleVar();

  ImGui::EndTable();

  ImGui::PopID();

}

void FLB::UiUtils::drawVec3Control(const std::string& label, glm::vec3& values, const glm::vec3& resetValues, float columnWidth)
{
  ImGuiIO& io = ImGui::GetIO();
  auto boldFont = io.Fonts -> Fonts[0];

  ImGui::PushID(label.c_str());

  ImGui::BeginTable("##", 2);
  ImGui::TableSetupColumn("##1", ImGuiTableColumnFlags_WidthFixed, columnWidth);
  ImGui::TableSetupColumn("##2", ImGuiTableColumnFlags_WidthFixed,  3 * columnWidth);

  ImGui::TableNextRow();
  ImGui::TableSetColumnIndex(0);
  ImGui::Text(label.c_str());
  ImGui::TableNextColumn();


  ImGui::PushMultiItemsWidths(3, ImGui::CalcItemWidth());
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0.0f, 0.0f});

  float lineHeight = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
  ImVec2 buttonSize = { lineHeight + 3.0f, lineHeight };

  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.8f, 0.1f, 0.15f, 1.0f });
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.9f, 0.2f, 0.2f, 1.0f });
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.8f, 0.1f, 0.15f, 1.0f });
  ImGui::PushFont(boldFont);
  if (ImGui::Button("X")) values.x = resetValues.x;
  ImGui::PopFont();
  ImGui::PopStyleColor(3);

  ImGui::SameLine();
  ImGui::DragFloat("##X", &values.x, 0.1f, 0.0f, 0.0f, "%.2f");
  ImGui::PopItemWidth();
  ImGui::SameLine();

  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f });
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.3f, 0.8f, 0.3f, 1.0f });
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f });  
  ImGui::PushFont(boldFont);
  if (ImGui::Button("Y")) values.y = resetValues.y;
  ImGui::PopFont();
  ImGui::PopStyleColor(3);

  ImGui::SameLine();
  ImGui::DragFloat("##Y", &values.y, 0.1f, 0.0f, 0.0f, "%.2f");
  ImGui::PopItemWidth();
  ImGui::SameLine();

  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.1f, 0.25f, 0.8f, 1.0f });
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.2f, 0.35f, 0.9f, 1.0f });
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.1f, 0.25f, 0.8f, 1.0f });
  ImGui::PushFont(boldFont);
  if (ImGui::Button("Z")) values.z = resetValues.z;
  ImGui::PopFont();
  ImGui::PopStyleColor(3);

  ImGui::SameLine();
  ImGui::DragFloat("##Z", &values.z, 0.1f, 0.0f, 0.0f, "%.2f");
  ImGui::PopItemWidth();

  ImGui::PopStyleVar();

  ImGui::EndTable();

  ImGui::PopID();
}

void FLB::UiUtils::drawComboBox(std::vector<std::string>& datasetsName, int& currentIdx, const char* name)
{
  std::string previewValue;
  if (datasetsName.size() > 0) previewValue = datasetsName[currentIdx]; 
  
  if (ImGui::BeginCombo(name, previewValue.c_str()))
  {
    for (int i = 0; i < datasetsName.size(); i++)
    {
      const bool isSelected = (currentIdx == i);
      if (ImGui::Selectable(datasetsName[i].c_str(), isSelected)) currentIdx = i;
      if (isSelected) ImGui::SetItemDefaultFocus();
    }
    ImGui::EndCombo();
  }
}

void FLB::UiUtils::helpMarker(const char* description)
{
  ImGui::TextDisabled("(?)");
  if (ImGui::BeginItemTooltip())
  {
    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
    ImGui::TextUnformatted(description);
    ImGui::PopTextWrapPos();
    ImGui::EndTooltip();
  }
}
