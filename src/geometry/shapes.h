#pragma once

#include <cstdint>

#include "../utils.h"

namespace FLB 
{
  class Shape
  {
    public:
      virtual ~Shape() = default;
      /**
       * @brief Change the node types of the domain inside the obstacle to the type indicated
       *
       * @param[in] h_flags  Flags of the domain
       *
       */
      virtual void initDomainShape(uint8_t* h_flags, FLB::TypesNodes typeNode) {return;};
      virtual bool isNodeInside(float xSI, float ySI, float zSI) {return 0;};

  };

  class CircleShape: public Shape
  {
    public:
      CircleShape(float radius, float thickness, float x, float y, float z);
      bool isNodeInside(float xSI, float ySI, float zSI) override;

    private:
      float x0, y0, z0; // Coordinates of the center (meters)
      float radius, thickness; // (meters)
      float angleRotation; // degrees
      unsigned int axisRotation; // 0 (xaxis), 1(yaxis), 2 ( 
  };

  class RectangleShape: public Shape
  {
    public:

      /**
       * @brief Constructor
       *
       * @param[in]  height height of the obstacle
       * @param[in]  xWidth
       * @param[in]
       * @param[in]
       * @param[in]
       *
       *
       */
      RectangleShape(float height, float xWidth, float zWidth, float x, float y, float z);

      bool isNodeInside(float xSI, float ySI, float zSI) override;

    private:
      const float x0, y0, z0; // Coordinates of the center (meters)
      const float height, xWidth, zWidth; // meters
      float angleRotation; // degrees
      unsigned int axisRotation; // 0 (xaxis), 1(yaxis), 2 ( 
  };

  enum class TypesCDW
  {
    CIRCLE = 0, RECTANGLE = 1
  };

  class CDW
  {
    public:
      TypesCDW typeCDW;

      virtual ~CDW() = default;

      /**
       * @brief It identifies if the node of the mesh is inside de CDW
       * 
       * @param[in]
       * @param[in]
       *
       */
      virtual bool isNodeInside(float yDiff, float zDiff) {return 0;};
      virtual void setValue(float value, int type = 0) {return;};
      virtual float getValue(int type = 0) {};
      //CDW() {};
      //virtual ~CDW() = 0;

  };

  class CircularCDW: public CDW
  {
    public:
      //float radius;

      bool isNodeInside(float yDiff, float zDiff) override;
      void setValue(float value, int type = 0) override;
      float getValue(int type = 0) override;

      private:
      float radius;
  };

  class RectangularCDW: public CDW
  {

    public:

      // height is in the y direction and width in the z direction
      // width is only used in a 3D case 
      float height, width;

      bool isNodeInside(float yDiff, float zDiff) override;
      void setValue(float value, int type) override;
      float getValue(int type) override;

  };
}
