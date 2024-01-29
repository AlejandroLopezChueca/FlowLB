#include "scene.h"
#include "graphics/renderer.h"
#include "entity.h"
#include "components.h"

#include <cstdint>
#include <iostream>

FLB::Scene::Scene(const FLB::Mesh* mesh)
  : m_ScalarVectorialFieldComponent(mesh -> getVertexArray()), m_NumPointsMesh(mesh -> getNumberPointsMesh())
{
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

  m_Registry.destroy(entity);
}

void FLB::Scene::update(const FLB::OrthographicCamera &camera)
{
  // Scalar/Vectorial fields
  FLB::Renderer::drawScalarVectorialField(m_ScalarVectorialFieldComponent, m_NumPointsMesh);

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
