#include <cmath>
#include <iostream>

#include "shapes.h"


FLB::CircleShape::CircleShape(float radius, float thickness, float x, float y, float z)
  : x0(x), y0(y), z0(z), radius(radius), thickness(thickness) {}

bool FLB::CircleShape::isNodeInside(float xSI, float ySI, float zSI)
{
  float distancePointsYZ = (ySI - y0)*(ySI - y0) + (zSI - z0)*(zSI - z0); 
  if (x0 - thickness/2 > xSI && x0 + thickness/2 < xSI && distancePointsYZ<= radius * radius) return true;
  return false;
}

FLB::RectangleShape::RectangleShape(float height, float xWidth, float zWidth, float x, float y, float z)
  :  x0(x), y0(y), z0(z), height(height), xWidth(xWidth), zWidth(zWidth) {}

bool FLB::RectangleShape::isNodeInside(float xSI, float ySI, float zSI)
{

}

bool FLB::CircularCDW::isNodeInside(float yDiff, float zDiff)
{
  if (yDiff*yDiff + zDiff*zDiff <= radius*radius) return true;
  return false;
}

void FLB::CircularCDW::setValue(float value, int type)
{
  radius = value;
}

float FLB::CircularCDW::getValue(int type)
{
  return radius;
}

bool FLB::RectangularCDW::isNodeInside(float yDiff, float zDiff)
{
  if (2*yDiff <= height && 2*zDiff <= width) return true;
  return false;
}

void FLB::RectangularCDW::setValue(float value, int type)
{
  if (type == 0) height = value;
  else width = value;
}

float FLB::RectangularCDW::getValue(int type)
{
  if (type == 0) return height;
  else return width;
}
