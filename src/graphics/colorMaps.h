#pragma once

#include "glm/ext/vector_float3.hpp"
#include <glm/glm.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace FLB::ColorMaps
{
// Scientific color maps from https://www.fabiocrameri.ch/colourmaps/
  extern std::array<std::string, 4> nameColorMaps;
  extern std::array<const std::vector<glm::vec3>, 4> colorMaps;
}
