#pragma once

#include "geometry/mesh.h"
#include "graphics/scene/componentsMesh.h"
#include "graphics/vertexArray.h"

#include <cstddef>
#include <glm/fwd.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_float4x4.hpp>
#include <glm/gtx/quaternion.hpp>
#include <memory>

namespace FLB
{
  struct DrawComponent
  {
    bool draw = true;

    DrawComponent() = default;
    DrawComponent(const DrawComponent&) = default;
  };

  struct TagComponent
  {
    std::string tag;

    TagComponent() = default;
    TagComponent(const std::string& str)
      :tag(str) {}
  };

  struct TransformComponent
  {
    glm::vec3 translation = {0.0f, 0.0f, 0.0f};
    glm::vec3 rotation = {0.0f, 0.0f, 0.0f};
    glm::vec3 scale = {1.0f, 1.0f, 1.0f};

    glm::mat4 identity = glm::mat4(1.0f);

    TransformComponent() = default;

    glm::mat4 getTransform() const
    {
      glm::mat4 rotationMatrix = glm::toMat4(glm::quat(rotation));
      return glm::translate(identity, translation) * rotationMatrix * glm::scale(identity, scale);
    }

    glm::mat4 getTransformNoScale() const
    {
      glm::mat4 rotationMatrix = glm::toMat4(glm::quat(rotation));
      return glm::translate(identity, translation) * rotationMatrix;
    }
    glm::mat4 getTransformNoRotation() const
    {
      return glm::translate(identity, translation) * glm::scale(identity, scale);
    }
    
    glm::mat4 getTranslationMatrix() const
    {
      return glm::translate(identity, translation);
    }

  };

  struct ScalarVectorialFieldComponent
  {
    glm::vec2 maxMinValues = {1.0f, 0.0f};
    int currentIdxColorMap = 0;
    int type = 0; // 0 for point and 1 for texture
    int idField = 0; // 0 Velocity, 1 Pressure
    float sizePoints = 1.0f;
    const FLB::VertexArray* meshVertexArray; // vertex array of the domain mesh
    
    ScalarVectorialFieldComponent() = default;
   
    ScalarVectorialFieldComponent(const FLB::VertexArray* vertexArray)
      : meshVertexArray(vertexArray)
    {
    }
  };

  struct StreamLinesComponent
  {

  };

  struct Arrow2DComponent
  {
    std::unique_ptr<FLB::Arrow2DMesh> mesh;
    glm::ivec2 numberPoints = {2, 2};
    float scale = 0.1f;
    bool updateMesh = false;
    int currentIdxColorMap = 0;
    glm::vec2 maxMinValues = {1.0f, 0.0f};

    const FLB::VertexArray* vertexArray;
    bool* draw;

    Arrow2DComponent(FLB::API api, Fl_Simple_Terminal* terminal, bool* drawComponent, const FLB::Mesh* meshDomain)
      : draw(drawComponent)
    {
      mesh = std::make_unique<FLB::Arrow2DMesh>(api, terminal, meshDomain);
      vertexArray = mesh -> getVertexArray();
    }
  };

  struct RectangleComponent
  {
    glm::vec4 color{1.0f, 0.0f, 0.0f, 1.0f};
    glm::vec2 dimensions = {1.0f, 1.0f};
    bool draw = false;
    int idx;

    RectangleComponent() = default;
  };

}
