
#include "glm/ext/vector_float3.hpp"
#include <glm/glm.hpp>
#include <string>
#include <unordered_map>
#include <vector>


namespace FLB
{
// Scientific color maps from https://www.fabiocrameri.ch/colourmaps/
  namespace ColorMaps
  {
    std::unordered_map<std::string, const std::vector<glm::vec3>> colorMaps =
    {
      {
	"imola",
	{
	  { 0.101f, 0.200f, 0.700f },
	  { 0.142f, 0.277f, 0.662f },
	  { 0.178f, 0.350f, 0.625f },
	  { 0.224f, 0.418f, 0.580f },
	  { 0.287f, 0.484f, 0.521f },
	  { 0.375f, 0.573f, 0.483f },
	  { 0.482f, 0.681f, 0.455f },
	  { 0.599f, 0.797f, 0.427f },
	  { 0.768f, 0.915f, 0.403f },
	  { 1.000f, 1.000f, 0.400f }
	}
      },
    };
  }
}
