#include "addPanel.h"

#include <imgui.h>

FLB::MetricsPanel::MetricsPanel(FLB::API api, FLB::Mesh* mesh, float& time)
  : m_NumPointsMesh(mesh ->getNumberPointsMesh()), m_Time(time)
{

}

void FLB::MetricsPanel::onImGuiRender(bool *open)
{
  ImGui::SetNextWindowSize(ImVec2(200,500), ImGuiCond_FirstUseEver);
  ImGui::Begin("Metrics", open);
  ImGuiIO& io = ImGui::GetIO();
  ImGui::Text("Application average: %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
  ImGui::Text("Time simulation: %.3f s", m_Time);
  ImGui::SeparatorText("Domain");
  ImGui::Text(" Number of points: %'lu ", m_NumPointsMesh);
  ImGui::End();

}
