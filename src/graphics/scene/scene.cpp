#include "scene.h"
#include "graphics/cameraController.h"
#include "graphics/renderer.h"
#include "entity.h"
#include "components.h"

#include <array>
#include <cstdint>
#include <iostream>

FLB::Scene::Scene(const FLB::Mesh* mesh)
  : m_ScalarVectorialFieldComponent(mesh -> getVertexArray()), m_NumPointsMesh(mesh -> getNumberPointsMesh())
{
  m_DrawIsosurface = &m_DefaultDrawIsosurface;
}

FLB::Entity FLB::Scene::createEntity(const std::string &name)
{
  FLB::Entity entity = FLB::Entity(m_Registry.create(), this);
  entity.addComponent<FLB::DrawComponent>();
  auto& tag = entity.addComponent<FLB::TagComponent>(name);
  tag.tag = name.empty() ? "Entity" : name;
  return entity;
}

void FLB::Scene::destroyEntity(FLB::Entity &entity)
{
  if (entity.hasComponent<FLB::RectangleComponent>()) 
  {
    // get idx of the entity to delete
    auto& rectangleComponentToDelete = entity.getComponent<FLB::RectangleComponent>();
    uint32_t idxToDelete = rectangleComponentToDelete.idx;

    // for the rest of the rectangles it is necessary to correct its idx if the idx is greater that the idx to delete
    auto view = m_Registry.view<FLB::RectangleComponent>();
    for (auto entityView : view)
    {
      auto& rectangleComponent = view.get<FLB::RectangleComponent>(entityView);
      if (rectangleComponent.idx > idxToDelete)
      {
	rectangleComponent.idx -= 1;
      }

    }
    FLB::Renderer::removeInstanceRectangles(idxToDelete);
  }
  
  // prevent indeterminate behaviour
  if (entity.hasComponent<FLB::IsoSurfaceComponent>()) 
  {
    // TODO Only valid for one isosurface 
    m_DrawIsosurface = &m_DefaultDrawIsosurface;
  }

  m_Registry.destroy(entity);
}

void FLB::Scene::update(FLB::OrthographicCameraController* cameraController)
{
  // Scalar/Vectorial fields
  if (m_DrawScalarVectorialField) FLB::Renderer::drawScalarVectorialField(m_ScalarVectorialFieldComponent, m_NumPointsMesh);

  if (*m_DrawIsosurface) 
  {
    bool updateIsoSurfaceBounds = cameraController -> getUpdateCameraDomainCornersSI();
    // Only calculate again if the camera is moved or the frameBuffer is resized
    const std::array<float, 8>& cameraDomainCornersSI = cameraController -> getCameraDomainBoundsSI(); // coordiantes of the doomain's corners that are view by the camera 
    auto view = m_Registry.view<FLB::IsoSurfaceComponent>();
    for (auto entity : view)
    {
      FLB::IsoSurfaceComponent& component = view.get<FLB::IsoSurfaceComponent>(entity);
      if (*component.draw) FLB::Renderer::drawIsoSurface(component, cameraDomainCornersSI, updateIsoSurfaceBounds);
    }
  }

  // Rectangles
  if (m_NumActiveRectangles)
  {
    if (*m_SelectedEntity)
    {
      auto group = m_Registry.group<FLB::RectangleComponent>(entt::get<FLB::TransformComponent>);
      auto [rectangleComponent, transformComponent] = group.get<FLB::RectangleComponent, FLB::TransformComponent>(*m_SelectedEntity);
    
      int idx = rectangleComponent.idx;
      glm::vec4& color = rectangleComponent.color;
      glm::mat4 transform = transformComponent.getTransform();
      FLB::Renderer::drawInstancedRectangles(idx, true, &color, &transform);
    }
    else FLB::Renderer::drawInstancedRectangles(0, false, nullptr, nullptr);
  }

  // Arrow 2D Glyph
  {
    auto group = m_Registry.group<FLB::Arrow2DComponent>(entt::get<FLB::TransformComponent>);

    if (group.size() > 0) FLB::Renderer::setArrowShader();

    for (auto entity : group)
    {
      auto [arrow2DComponent, transformComponent] = group.get<FLB::Arrow2DComponent, FLB::TransformComponent>(entity);
      
      if (*arrow2DComponent.draw) FLB::Renderer::drawInstancedArrows(arrow2DComponent, transformComponent);
    }
  }

}
