#pragma once

#include "components.h"
#include "geometry/mesh.h"
#include "graphics/camera.h"

#include "entt.hpp"
#include <cstddef>
#include <cstdint>
#include <string>

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

      void update(const FLB::OrthographicCamera& camera);

      const int* getRenderingScalarVectorialField() const { return &m_ScalarVectorialFieldComponent.type;}

    private:
      entt::registry m_Registry;
      FLB::ScalarVectorialFieldComponent m_ScalarVectorialFieldComponent;

      FLB::Entity* m_SelectedEntity;
      size_t m_NumPointsMesh;
      uint32_t m_NumActiveRectangles = 0;

      friend class HierarchyPanel;
      friend class Entity;


  };


}
