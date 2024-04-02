#pragma once

#include "graphics/cameraController.h"
#include <array>
#include <string>

namespace FLB 
{
  class RenderLayer;

  class SettingsPanel2D
  {
    public:
      SettingsPanel2D(FLB::OrthographicCameraController* cameraController, FLB::RenderLayer* renderLayer);
      void onImGuiRender(bool* open);

      private:
	void showCameraSettings();
	void showRenderingSettings();

	FLB::OrthographicCameraController* m_CameraController;
	FLB::RenderLayer* m_RenderLayer;

	const char* m_Items[2] = {"Camera", "Rendering"};
  };

}
