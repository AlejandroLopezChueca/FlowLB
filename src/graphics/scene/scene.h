#pragma once

#include "components.h"
#include "geometry/mesh.h"
#include "graphics/API.h"
#include "graphics/camera.h"

#include "entt.hpp"
#include "graphics/cameraController.h"
#include "graphics/texture.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace FLB 
{
  class Entity;

  class Scene
  {
    public:
      Scene() = default;
      Scene(const FLB::Mesh* mesh);
      ~Scene() = default;

      FLB::Entity createEntity(const std::string& name);

      void destroyEntity(FLB::Entity& entity);

      void update(FLB::OrthographicCameraController* cameraController);

      void setOrthographicCameraController(FLB::OrthographicCameraController* orthographicCameraController) {m_OrthographicCameraController = orthographicCameraController;}
      std::array<float, 8>& getOrthographicCameraDomainBoundsSI() { return m_OrthographicCameraController -> getCameraDomainBoundsSI();}

      const entt::registry& getRegistry() const {return m_Registry;}

      const int* getRenderingScalarVectorialField() const { return &m_ScalarVectorialFieldComponent.type;}
      bool& getDrawScalarVectorialField() {return m_DrawScalarVectorialField;}
      
      const bool* isIsosurfaceRendering() const {return m_DrawIsosurface;}
      void setIsosurfaceRendering(const bool* drawIsoEntity) {m_DrawIsosurface = drawIsoEntity;}

      void setCalculationAPI(const FLB::CalculationAPI calculationAPI) {m_CalculationAPI = calculationAPI;}

      const FLB::ScalarVectorialFieldComponent& getScalarVectorialFieldComponent() const {return m_ScalarVectorialFieldComponent;}

    private:
      entt::registry m_Registry;
      FLB::ScalarVectorialFieldComponent m_ScalarVectorialFieldComponent;
      FLB::OrthographicCameraController* m_OrthographicCameraController;
      FLB::CalculationAPI m_CalculationAPI;

      FLB::Entity* m_SelectedEntity;
      size_t m_NumPointsMesh;
      uint32_t m_NumActiveRectangles = 0;
      bool m_DrawScalarVectorialField = true;
      
      const bool* m_DrawIsosurface;
      bool m_DefaultDrawIsosurface = false; // dafault value for initialization and when the isosurface is destroyed 

      friend class HierarchyPanel;
      friend class Entity;
  };
}
