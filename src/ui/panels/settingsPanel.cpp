#include "settingsPanel.h"
#include "graphics/cameraController.h"
#include "ui/renderLayer.h"
#include "ui/uiUtils.h"

#include <imgui.h>

FLB::SettingsPanel2D::SettingsPanel2D(FLB::OrthographicCameraController* cameraController, FLB::RenderLayer* renderLayer)
  : m_CameraController(cameraController), m_RenderLayer(renderLayer) {}

void FLB::SettingsPanel2D::onImGuiRender(bool *open)
{
  ImGui::Begin("Settings", open);
  // Left child
  static int selectedSetting = -1;
  ImGui::BeginChild("leftSetting", ImVec2(150, 0), true);

  for (int i = 0; i < 2; i++)
  {
    if (ImGui::Selectable(m_Items[i], selectedSetting == i)) selectedSetting = i;
  }
  ImGui::EndChild();

  // Rigth child
  ImGui::SameLine();
  ImGui::BeginChild("rightSetting");
  if (selectedSetting == 0) showCameraSettings();
  else if(selectedSetting == 1) showRenderingSettings();
  ImGui::EndChild();

  ImGui::End();
}

void FLB::SettingsPanel2D::showCameraSettings()
{
  ImGui::SeparatorText("Speed");
  ImGui::InputFloat("Translation", &m_CameraController -> getCameraTranslationSpeed(), 0.1f, 2.0f, "%.1f");
}

void FLB::SettingsPanel2D::showRenderingSettings()
{
  ImGui::SeparatorText("Rendering");
  ImGui::SliderInt("Framerate Limit", &m_RenderLayer -> m_FramerateLimit, 15, 120);
  ImGui::SameLine();
  FLB::UiUtils::helpMarker("This limit is only exact when the calculation is stopped, if it is running, the real FPS will be lower");
  m_RenderLayer -> m_SecondsFrameRate = 1.0f /((float)m_RenderLayer -> m_FramerateLimit);
}
