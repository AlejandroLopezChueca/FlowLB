#pragma once

#include "geometry/mesh.h"

#include <cstdint>
#include <glm/glm.hpp>
#include <string>
#include <system_error>
#include <vector>
#include <charconv>

namespace FLB::Math
{
  bool decomposeTransform(const glm::mat4& transform, glm::vec3& translation, glm::vec3& rotation, glm::vec3& scale);
 
  template<typename T>
  bool essentiallyEqual(T a, T b, T epsilon)
  {
    return std::abs(a - b) <= epsilon;
  }

  double distance2DPoints(double x1, double x2, double y1, double y2);
  
  double distanceSquaredPointLine(double x, double y, double z, std::vector<FLB::Point>& points, uint32_t idxEndPointBase, double increment);

  template<typename T>
  bool convertStringToDecimal(T& number, std::string& str)
  {
    auto[ptr, ec] = std::from_chars(str.data(), str.data() + str.size(), number, std::chars_format::general);
    return ec == std::errc();
  }

}
