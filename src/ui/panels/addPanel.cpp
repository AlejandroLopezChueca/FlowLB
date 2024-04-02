#include "addPanel.h"
#include "cuda/cudaUtils.cuh"
#include "glm/fwd.hpp"
#include "graphics/cameraController.h"
#include "graphics/scene/components.h"
#include "graphics/scene/entity.h"
#include "ui/uiUtils.h"

#include <array>
#include <cmath>
#include <imgui.h>
#include <iostream>
#include <vector>


//////////////////////////////// MetricsPanel ////////////////////////////////
FLB::MetricsPanel::MetricsPanel(FLB::API api, FLB::Mesh* mesh, float& time, float& miliSecondsSimulation, float& frameRateSimulation)
  : m_NumPointsMesh(mesh ->getNumberPointsMesh()), m_Time(time), m_MiliSecondsSimulation(miliSecondsSimulation), m_FrameRateSimulation(frameRateSimulation)
{

}

void FLB::MetricsPanel::onImGuiRender(bool *open)
{
  ImGui::SetNextWindowSize(ImVec2(200,500), ImGuiCond_FirstUseEver);
  ImGui::Begin("Metrics", open);
  ImGui::SeparatorText("Time");
  ImGui::Text("Calculation average: %.3f ms/iteration (%.1f IPS)", m_MiliSecondsSimulation, m_FrameRateSimulation);
  ImGuiIO& io = ImGui::GetIO();
  ImGui::Text("Graphics average: %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
  ImGui::Text("Time simulation: %.3f s", m_Time);
  ImGui::SeparatorText("Domain");
  ImGui::Text(" Number of points: %'lu ", m_NumPointsMesh);
  ImGui::SeparatorText("Memory");
  ImGui::Text("Used Memory GPU (MB): %.1f\n", m_UsedMemory);
  ImGui::Text("Free Memory GPU (MB): %.1f\n", m_FreeMemory);
  ImGui::End();
}

//////////////////////////////// IsosurfacePanel ////////////////////////////////

FLB::IsosurfacePanel::IsosurfacePanel(FLB::API api, FLB::CalculationAPI calculationApi, Fl_Simple_Terminal* terminal, const FLB::Mesh* mesh)
  : m_Api(api), m_CalculationApi(calculationApi), m_Terminal(terminal), m_DomainMesh(mesh)
{

}

void FLB::IsosurfacePanel::onImGuiRender(bool *open)
{
  ImGui::SetNextWindowSize(ImVec2(500,200), ImGuiCond_FirstUseEver);
  ImGui::Begin("Isosurface", open);

  ImGui::SeparatorText("Texture size");
  ImGui::DragInt("Width (pix)", &m_Width);
  ImGui::DragInt("Height (pix)", &m_Height);


  // Button add
  ImGui::Dummy(ImVec2(0.0f, 20.0f));
  float width = ImGui::GetWindowSize().x;
  float height = ImGui::GetWindowSize().y;
  ImVec2 buttonSize = {45, 30};
  ImGui::SetCursorPos(ImVec2((width - 1.3f * buttonSize.x), (height - 1.65f * buttonSize.y)));
  if (ImGui::Button("Add", buttonSize))
  {
    createIsosurface();
  }
  ImGui::End();
}

void FLB::IsosurfacePanel::createIsosurface()
{
  // For now, there is only one Isosurface allowed
  auto view = m_Scene -> getRegistry().view<FLB::IsoSurfaceComponent>();
  if (view.size() > 0) return;

  FLB::Entity isoEntity = m_Scene -> createEntity("Isosurface");
  
  auto& drawComponent = isoEntity.getComponent<FLB::DrawComponent>();
  if (m_DomainMesh ->is3D())
  {

  }
  else 
  {
    std::array<float, 8>& vertices  = m_Scene -> getOrthographicCameraDomainBoundsSI();
    isoEntity.addComponent<FLB::IsoSurfaceComponent>(m_Api, m_Terminal, &drawComponent.draw, m_DomainMesh, m_Width, m_Height, vertices);
  }
    
  auto& isosurfaceComponent = isoEntity.getComponent<FLB::IsoSurfaceComponent>();

  // save pointers of the component and draw action in the scene
  m_Scene -> setIsosurfaceRendering(isosurfaceComponent.draw);

  if (m_CalculationApi == FLB::CalculationAPI::CUDA)
  {
    unsigned int flags = cudaGraphicsRegisterFlagsWriteDiscard | cudaGraphicsRegisterFlagsSurfaceLoadStore;
    FLB::CudaUtils::mapTexture2DToOpenGL(&isosurfaceComponent.cudaResTextureIsoSurface, isosurfaceComponent.texture, flags);
  }
}

//////////////////////////////// ConsultValuesPanel ////////////////////////////////

FLB::ConsultValuesPanel::ConsultValuesPanel(const FLB::Texture2D* texture, const FLB::Window* window, const FLB::OrthographicCameraController* cameraController, const int& idxVectorRepresentation, const int& idScalarVectorField)
  : m_ScalarVectorialFieldsTexture(texture), m_RenderWindow(window), m_CameraController(cameraController), m_CurrentIdxVectorRepresentation(idxVectorRepresentation), m_IdScalarVectorField(idScalarVectorField)
{

}

void FLB::ConsultValuesPanel::onImGuiRender(bool *open, const glm::vec2* viewportBounds)
{
  ImGui::SetNextWindowSize(ImVec2(400,200), ImGuiCond_FirstUseEver);
  ImGui::Begin("Consult values", open);

  if (m_RenderWindow -> isLeftButtonMouseClicked()) getValue(viewportBounds);
  
  ImGui::InputFloat("Value", &m_ConsultedValue, 0.0f, 0.0f, "%.3f", ImGuiInputTextFlags_ReadOnly);
  ImGui::End();
}

void FLB::ConsultValuesPanel::getValue(const glm::vec2* viewportBounds)
{
  ImVec2 mousePos = ImGui::GetMousePos();
  mousePos.x -= viewportBounds[0].x;
  mousePos.y -= viewportBounds[0].y;
  const glm::vec2 viewportSize = viewportBounds[1] - viewportBounds[0];
  mousePos.y = viewportSize.y - mousePos.y; // flip y coord to match OpenGL texture coordinates

  // get values from camera to calculate distances in view space
  const glm::vec4& viewUpperLeftCornerDomain = m_CameraController -> getViewUpperLeftCornerDomain();
  const glm::vec4& viewLowerRightCornerDomain = m_CameraController -> getViewLowerRightCornerDomain();
  const float xMaxViewCamera = m_CameraController -> getxMaxViewCamera(); 
  const float yMaxViewCamera = m_CameraController -> getyMaxViewCamera();

  // calculate max and min pixeles of the domain in pixels of the viewport
  const float xMinPixelDomain = viewUpperLeftCornerDomain.x > (-xMaxViewCamera) ? (viewUpperLeftCornerDomain.x + xMaxViewCamera) / (2 * xMaxViewCamera) * viewportSize.x : 0;
  const float xMaxPixelDomain = viewLowerRightCornerDomain.x < xMaxViewCamera ? (viewLowerRightCornerDomain.x + xMaxViewCamera) / (2 * xMaxViewCamera) * viewportSize.x : viewportSize.x;

  const float yMinPixelDomain = viewLowerRightCornerDomain.y > (-yMaxViewCamera) ? (viewLowerRightCornerDomain.y + yMaxViewCamera) / (2 * yMaxViewCamera) * viewportSize.y : 0;
  const float yMaxPixelDomain = viewUpperLeftCornerDomain.y < yMaxViewCamera ? (viewUpperLeftCornerDomain.y + yMaxViewCamera) / (2 * yMaxViewCamera) * viewportSize.y : viewportSize.y;

  // only if it is inside the texture 
  if (mousePos.x >= xMinPixelDomain && mousePos.x <= xMaxPixelDomain && mousePos.y >= yMinPixelDomain && mousePos.y <= yMaxPixelDomain)
  {
    // calculate offset from origin of texture in pixels of the viewport
    const float xOffset = mousePos.x - xMinPixelDomain;
    const float yOffset = mousePos.y - yMinPixelDomain;

    // calculate width and height of the texture in viewport pixels
    const float viewWidth = (viewLowerRightCornerDomain.x - viewUpperLeftCornerDomain.x) / (2 * xMaxViewCamera) * viewportSize.x;
    const float viewHeight = (viewUpperLeftCornerDomain.y - viewLowerRightCornerDomain.y) / (2 * yMaxViewCamera) * viewportSize.x;

    // get position of the pixels in the texture
    const int xPos = viewUpperLeftCornerDomain.x > (-xMaxViewCamera) ? xOffset / viewWidth *  m_ScalarVectorialFieldsTexture -> getWidth() : ((-xMaxViewCamera - viewUpperLeftCornerDomain.x) / (2 * xMaxViewCamera) * viewportSize.x + xOffset) / viewWidth * m_ScalarVectorialFieldsTexture -> getWidth();
    const int yPos = viewLowerRightCornerDomain.y > (-yMaxViewCamera) ? yOffset / viewHeight *  m_ScalarVectorialFieldsTexture -> getHeight() : ((-yMaxViewCamera - viewLowerRightCornerDomain.y) / (2 * yMaxViewCamera) * viewportSize.y + yOffset) / viewHeight * m_ScalarVectorialFieldsTexture -> getHeight();

    if (m_IdScalarVectorField == 0) // velocity
    {
      float data[2];
      m_ScalarVectorialFieldsTexture -> readPixels(xPos, yPos, 1, 1, 2 * sizeof(float), data);
      switch (m_CurrentIdxVectorRepresentation)
      {
	case 0: {m_ConsultedValue = std::sqrt(data[0] * data[0] + data[1] * data[1]); break;} // Magnitude   
	case 1: {m_ConsultedValue = data[0]; break;} // x component
	case 2: {m_ConsultedValue = data[1]; break;} // y component   
      }
    }
  }
}

