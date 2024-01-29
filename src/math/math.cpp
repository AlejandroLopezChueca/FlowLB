#include "math.h"

#include <cmath>
#include <cstdint>
#include <numeric>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/matrix_decompose.hpp>

bool FLB::Math::decomposeTransform(const glm::mat4 &transform, glm::vec3 &translation, glm::vec3 &rotation, glm::vec3 &scale)
{
    // From glm::decompose in matrix_decompose.inl
    using namespace glm;
    using T = float;

    mat4 LocalMatrix(transform);

    // Normalize the matrix.
    if (epsilonEqual(LocalMatrix[3][3], static_cast<float>(0), epsilon<T>()))
	    return false;

    // First, isolate perspective.  This is the messiest.
    if (
	  epsilonNotEqual(LocalMatrix[0][3], static_cast<T>(0), epsilon<T>()) ||
	  epsilonNotEqual(LocalMatrix[1][3], static_cast<T>(0), epsilon<T>()) ||
	  epsilonNotEqual(LocalMatrix[2][3], static_cast<T>(0), epsilon<T>()))
    {
	    // Clear the perspective partition
	  LocalMatrix[0][3] = LocalMatrix[1][3] = LocalMatrix[2][3] = static_cast<T>(0);
	  LocalMatrix[3][3] = static_cast<T>(1);
    }

    // Next take care of translation (easy).
    translation = vec3(LocalMatrix[3]);
    LocalMatrix[3] = vec4(0, 0, 0, LocalMatrix[3].w);

    vec3 Row[3], Pdum3;

    // Now get scale and shear.
    for (length_t i = 0; i < 3; ++i)
	    for (length_t j = 0; j < 3; ++j)
		    Row[i][j] = LocalMatrix[i][j];

    // Compute X scale factor and normalize first row.
    scale.x = length(Row[0]);
    Row[0] = detail::scale(Row[0], static_cast<T>(1));
    scale.y = length(Row[1]);
    Row[1] = detail::scale(Row[1], static_cast<T>(1));
    scale.z = length(Row[2]);
    Row[2] = detail::scale(Row[2], static_cast<T>(1));

		// At this point, the matrix (in rows[]) is orthonormal.
		// Check for a coordinate system flip.  If the determinant
		// is -1, then negate the matrix and the scaling factors.
#if 0
		Pdum3 = cross(Row[1], Row[2]); // v3Cross(row[1], row[2], Pdum3);
		if (dot(Row[0], Pdum3) < 0)
		{
			for (length_t i = 0; i < 3; i++)
			{
				scale[i] *= static_cast<T>(-1);
				Row[i] *= static_cast<T>(-1);
			}
		}
#endif

    rotation.y = asin(-Row[0][2]);
    if (cos(rotation.y) != 0) 
    {
      rotation.x = atan2(Row[1][2], Row[2][2]);
      rotation.z = atan2(Row[0][1], Row[0][0]);
    }
    else 
    {
      rotation.x = atan2(-Row[2][0], Row[1][1]);
      rotation.z = 0;
    }

    return true;
}

double FLB::Math::distance2DPoints(double x1, double x2, double y1, double y2)
{
  double side1 = (x1 - x2);
  double side2 = (y1 - y2);
  return std::sqrt(side1 * side1 + side2 * side2);
}


double FLB::Math::distanceSquaredPointLine(double x, double y, double z, std::vector<FLB::Point> &points, uint32_t idxEndPointBase, double increment)
{
  // the distance is calculated with the vectorial product of the vector director of the line (v) and the vector PQ, Q = point in analysis and P= point of the line -> d = |QP x v| / |v|

  // use Point struct to define the vectors PQ and v
  // Sum increment to elevate the line, for example, to connect the centers of the CDW
  FLB::Point PQ{x - points[idxEndPointBase].x, y - (points[idxEndPointBase].y + increment), z - points[idxEndPointBase].z};
  FLB::Point v{points[idxEndPointBase - 1].x - points[idxEndPointBase].x, points[idxEndPointBase - 1].y - points[idxEndPointBase].y, points[idxEndPointBase - 1].z - points[idxEndPointBase].z};

  // vectorial product
  FLB::Point PQxv{PQ.y * v.z - PQ.z * v.y, PQ.z * v.x - PQ.x * v.z, PQ.x * v.y - PQ.y * v.x};

  // calculate distance squared
  return (PQxv.x * PQxv.x + PQxv.y * PQxv.y + PQxv.z * PQxv.z) / (v.x * v.x + v.y * v.y + v.z * v.z);
}
