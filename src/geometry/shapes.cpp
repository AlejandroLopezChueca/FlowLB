#include "shapes.h"
#include "io/csvReader.h"
#include "math/math.h"
#include "geometry/mesh.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <utility>
#include <vector>


FLB::Shape::Shape(double x, double y, double z)
  : m_x0(x), m_y0(y), m_z0(z) {}

////////////////////////////////////// CircleShape //////////////////////////////////////

FLB::CircleShape::CircleShape(double radius, double thickness, double x, double y, double z)
  : Shape(x, y, z), m_Radius(radius), m_Thickness(thickness) {}

bool FLB::CircleShape::isNodeInside(double xSI, double ySI, double zSI)
{
  // Plane Circle in the YZ plane
  if (m_Is3D)
  { 
    double xdistance = std::abs(m_x0 - xSI);
    double distancePointsYZ = (ySI - m_y0)*(ySI - m_y0) + (zSI - m_z0)*(zSI - m_z0); 
    if (xdistance > (m_Thickness/2.0 + m_Epsilon) || distancePointsYZ > (m_Radius * m_Radius + m_Epsilon)) return false;
  }
  else 
  {
    double distancePointsXY = (ySI - m_y0)*(ySI - m_y0) + (xSI - m_x0)*(xSI - m_x0);
    if (distancePointsXY > (m_Radius * m_Radius + m_Epsilon)) return false;
  }
  return true;
}

void FLB::CircleShape::setValue(const double value, FLB::TypeValue typeValue)
{
  switch (typeValue) 
  {
    case FLB::TypeValue::Radius: {m_Radius = value; break;}
    case FLB::TypeValue::Thickness: {m_Thickness = value; break;}
    case FLB::TypeValue::x0: {m_x0 = value; break;}
    case FLB::TypeValue::y0: {m_y0 = value; break;}
    case FLB::TypeValue::z0: {m_z0 = value; break;}
    default: break; 
  };
}

////////////////////////////////////// RectangleShape //////////////////////////////////////

FLB::RectangleShape::RectangleShape(double height, double xWidth, double zWidth, double x, double y, double z)
  :  Shape(x, y, z), m_Height(height), m_xWidth(xWidth), m_zWidth(zWidth) {}

bool FLB::RectangleShape::initShape(const std::filesystem::path& directoryPath, Fl_Simple_Terminal* terminal)
{
  m_RotatePoints = !FLB::Math::essentiallyEqual<double>(m_zRotation, 0.0, 1e-6);
  
  // creation of the rotation matrix
  // TODO fix to 3D rotation
  m_xRotationMatrix[0] = std::cos(-m_zRotation);
  m_xRotationMatrix[1] = std::sin(-m_zRotation);
  m_yRotationMatrix[0] = std::sin(-m_zRotation);
  m_yRotationMatrix[1] = std::cos(-m_zRotation);

  // max values without rotation
  double xMax = m_x0 + m_xWidth/2.0;
  double xMin = m_x0 - m_xWidth/2.0;
  double yMax = m_y0 + m_Height/2.0;
  double yMin = m_y0 - m_Height/2.0;
  double zMax = m_z0 + m_zWidth/2.0;
  double zMin = m_z0 - m_zWidth/2.0;
  if (m_RotatePoints)
  {
    // TODO  correct for 3D
    if (m_Is3D)
    {

    }
    else
    {

    }
  }
  else 
  {
    // TODO  correct for 3D
    if (m_Is3D)
    {
    }
    else 
    {
      m_xCoordinates.insert(m_xCoordinates.end(), {xMin, xMax, xMax, xMin});
      m_xCoordinates.insert(m_xCoordinates.end(), {yMin, yMin, yMax, yMax});
    }
  }
  // Epsilon to prevent decimal errors in checking bounding box if point is in edge
  m_xMax = xMax + m_Epsilon;
  m_xMin = xMin - m_Epsilon;
  m_yMax = yMax + m_Epsilon;
  m_yMin = yMin - m_Epsilon;
  m_zMax = zMax + m_Epsilon;
  m_zMin = zMin - m_Epsilon;

  return true;
}

bool FLB::RectangleShape::isNodeInside(double xSI, double ySI, double zSI)
{
  // if there is rotation, instead of rotate the rectangle, the point is rotate
  // TODO FIX to 3D rotation
  if (m_RotatePoints)
  {
    // move point to center of the rectangle as origin
    double prevX = (xSI - m_x0);
    double prevY = (ySI - m_y0);
    xSI = prevX * m_xRotationMatrix[0] - prevY * m_xRotationMatrix[1];
    ySI = prevX * m_yRotationMatrix[0] + prevY * m_yRotationMatrix[1];
    xSI += m_x0;
    ySI += m_y0;
    
    if (xSI > m_xMax || xSI < m_xMin || ySI > m_yMax || ySI < m_yMin || zSI > m_zMax || zSI < m_zMin) return false;
  }

  else if (xSI > m_xMax || xSI < m_xMin || ySI > m_yMax || ySI < m_yMin || zSI > m_zMax || zSI < m_zMin) return false;

  return true;
}

void FLB::RectangleShape::setValue(const double value, FLB::TypeValue typeValue)
{
  switch (typeValue) 
  {
    case FLB::TypeValue::Height: {m_Height = value; break;}
    case FLB::TypeValue::xWidth: {m_xWidth = value; break;}
    case FLB::TypeValue::zWidth: {m_zWidth = value; break;}
    case FLB::TypeValue::x0: {m_x0 = value; break;}
    case FLB::TypeValue::y0: {m_y0 = value; break;}
    case FLB::TypeValue::z0: {m_z0 = value; break;}
    case FLB::TypeValue::xRotation: {m_xRotation = value; break;}
    case FLB::TypeValue::yRotation: {m_yRotation = value; break;}
    case FLB::TypeValue::zRotation: {m_zRotation = value; break;}
    default: break; 
  }
}

////////////////////////////////////// Imported2DShape //////////////////////////////////////

bool FLB::Imported2DShape::initShape(const std::filesystem::path& directoryPath, Fl_Simple_Terminal* terminal)
{
  std::filesystem::path path = directoryPath.string() +"/" + m_Filename;

  auto [csvReader, err] = createCSVReader(path, terminal);
  if (err) return false;
  
  uint32_t posLine = 1;
  for (const std::vector<std::string>& row : csvReader)
  {
    double x, y;
    std::string xStr = row[0];
    if(!FLB::Math::convertStringToDecimal<double>(x, xStr)) 
    {
      terminal -> printf("Error converting the x coordinate in the line %u of the file: %s\n", posLine, path.c_str());
      return false;
    }
    
    std::string yStr = row[1];
    if(!FLB::Math::convertStringToDecimal<double>(y, yStr)) 
    {
      terminal -> printf("Error converting the y coordinate in the line %u of the file: %s\n", posLine, path.c_str());
      return false;
    }

    m_xCoordinates.push_back(x);
    m_yCoordinates.push_back(y);

    posLine += 1;
  }

  calculateAreaCentroid();
  moveRotateCoordinates();

  // to prevent decimal erros in checking bounding box if point is in edge
  m_xMax += m_Epsilon;
  m_xMin -= m_Epsilon;
  m_yMax += m_Epsilon;
  m_yMin -= m_Epsilon;

  return true;
}

bool FLB::Imported2DShape::isNodeInside(double xSI, double ySI, double zSI)
{
  // first check bounding box
  if (xSI > m_xMax || xSI < m_xMin || ySI > m_yMax || ySI < m_yMin) return false;

  // Crossing number method
  // Code from https://wrfranklin.org/Research/Short_Notes/pnpoly.html
  int i, j;
  bool isInside = false;
  for (i = 0, j = m_xCoordinates.size() - 1; i < m_xCoordinates.size(); j = i++)
  {
    if (((m_yCoordinates[i] > ySI) != (m_yCoordinates[j] > ySI)) && (xSI < (m_xCoordinates[j] - m_xCoordinates[i]) * (ySI - m_yCoordinates[i]) / (m_yCoordinates[j] - m_yCoordinates[i]) + m_xCoordinates[i])) isInside = !isInside;
  }

  return isInside;

}

void FLB::Imported2DShape::setValue(const double value, TypeValue typeValue)
{
  switch (typeValue)
  {
    case FLB::TypeValue::x0: {m_x0 = value; break;}
    case FLB::TypeValue::y0: {m_y0 = value; break;}
    case FLB::TypeValue::zRotation: {m_zRotation = value; break;}
    default: break;
  }
}

std::pair<FLB::CSVReader, bool> FLB::Imported2DShape::createCSVReader(const std::filesystem::path& path, Fl_Simple_Terminal* terminal)
{
  bool err = false;
  FLB::CSVReader csvReader{path.string()};
  if (!csvReader.isOpen())
  {
    terminal -> printf("Error opening the file: %s\n", path.c_str());
    err = true;
  }

  if (csvReader.isEmpty() && !err)
  {
    terminal -> printf("Error, the file %s is empty\n", path.c_str());
    err = true;
  }

  uint32_t minNumberLines = m_Is3D ? 4 : 3;
  if (!csvReader.checkMinNumberLines(minNumberLines) && !err)
  {
    terminal -> printf("Error, the file %s doesn't have the minimun nmuber of lines with data\n", path.c_str());
    err = true;
  }
  return {std::move(csvReader), err};
}

void FLB::Imported2DShape::calculateAreaCentroid()
{
  size_t N = m_xCoordinates.size();
  for (int i = 0; i < N - 1; i++)
  {
    double product = m_xCoordinates[i] * m_yCoordinates[i + 1] - m_xCoordinates[i + 1] * m_yCoordinates[i];
    m_Area += product;
    m_xCentroid += (m_xCoordinates[i] + m_xCoordinates[i + 1]) * product;
    m_yCentroid += (m_yCoordinates[i] + m_yCoordinates[i + 1]) * product;
  }
  // the last iteration use the first coordinate
  double product = (m_xCoordinates[N - 1] * m_yCoordinates[0] - m_xCoordinates[0] * m_yCoordinates[N - 1]);
  m_Area += product;
  m_xCentroid += (m_xCoordinates[N - 1] + m_xCoordinates[0]) * product;
  m_yCentroid += (m_yCoordinates[N - 1] + m_yCoordinates[0]) * product;
 
  m_Area *= 0.5;
  m_Area = m_Area < 0 ? -m_Area : m_Area; 

  m_xCentroid /= (6.0 * m_Area);
  m_xCentroid = m_xCentroid < 0 ? -m_xCentroid : m_xCentroid;
  m_yCentroid /= (6.0 * m_Area);
  m_yCentroid = m_yCentroid < 0 ? -m_yCentroid : m_yCentroid;
}

void FLB::Imported2DShape::moveRotateCoordinates()
{
  bool rotatePoints = !FLB::Math::essentiallyEqual<double>(m_zRotation, 0.0, 1e-6);

  double cosAlphaZ = std::cos(m_zRotation);
  double sinAlphaZ = std::sin(m_zRotation);
  for (int i = 0; i < m_xCoordinates.size(); i++)
  {
    //double x, y;
    double& x = m_xCoordinates[i];
    double& y = m_yCoordinates[i];

    // move around the geometric center to rotate
    x -= m_xCentroid;
    y -= m_yCentroid;

    // rotate
    if (rotatePoints)
    {
      double prevX = x;
      x = x * cosAlphaZ - y * sinAlphaZ;
      y = prevX * sinAlphaZ + y * cosAlphaZ;
    }

    // move to global position
    x += m_x0;
    y += m_y0;

    // max and min values for bouding box
    if (x > m_xMax) m_xMax = x;
    else if (x < m_xMin) m_xMin = x;
    if (y > m_yMax) m_yMax = y;
    else if (y < m_yMin) m_yMin = y; 
  }
}

////////////////////////////////////// Imported3DShape //////////////////////////////////////

bool FLB::Imported3DShape::initShape(const std::filesystem::path& directoryPath, Fl_Simple_Terminal* terminal)
{
  std::filesystem::path path = directoryPath.string() +"/" + m_Filename;

  auto [csvReader, err] = createCSVReader(path, terminal);
  if (err) return false;
  
  uint32_t posLine = 1;
  for (const std::vector<std::string>& row : csvReader)
  {
    double x, y, z;
    std::string xStr = row[0];
    if(!FLB::Math::convertStringToDecimal<double>(x, xStr)) 
    {
      terminal -> printf("Error converting the x coordinate in the line %u of the file: %s\n", posLine, path.c_str());
      return false;
    }
    
    std::string yStr = row[1];
    if(!FLB::Math::convertStringToDecimal<double>(y, yStr)) 
    {
      terminal -> printf("Error converting the y coordinate in the line %u of the file: %s\n", posLine, path.c_str());
      return false;
    }
    
    std::string zStr = row[2];
    if(!FLB::Math::convertStringToDecimal<double>(z, zStr))
    {
      terminal -> printf("Error converting the z coordinate in the line %u of the file: %s\n", posLine, path.c_str());
      return false;
    }

    m_xCoordinates.push_back(x);
    m_yCoordinates.push_back(y);
    m_zCoordinates.push_back(z);
  
    posLine += 1;
  }

  moveRotateCoordinates();

  return true;
}

bool FLB::Imported3DShape::isNodeInside(double xSI, double ySI, double zSI)
{
  if(xSI > m_xMax || xSI < m_xMin || ySI > m_yMax || ySI < m_xMin) return false;

  return true;
}

void FLB::Imported3DShape::setValue(const double value, TypeValue typeValue)
{
  switch (typeValue)
  {
    case FLB::TypeValue::x0: {m_x0 = value; break;}
    case FLB::TypeValue::y0: {m_y0 = value; break;}
    case FLB::TypeValue::z0: {m_z0 = value; break;}
    case FLB::TypeValue::xRotation: {m_xRotation = value; break;}
    case FLB::TypeValue::yRotation: {m_yRotation = value; break;}
    case FLB::TypeValue::zRotation: {m_zRotation = value; break;}
    default: break;
  }
}

void FLB::Imported3DShape::moveRotateCoordinates()
{
  bool rotatePoints = !FLB::Math::essentiallyEqual<double>(m_xRotation, 0.0, 1e-6) || !FLB::Math::essentiallyEqual<double>(m_yRotation, 0.0, 1e-6) || !FLB::Math::essentiallyEqual<double>(m_zRotation, 0.0, 1e-6);
  double cosAlphaX = std::cos(m_xRotation);

  double senAlphaX = std::sin(m_xRotation);
  for (int i = 0; i < m_xCoordinates.size(); i++)
  {
    double x, y, z;
    
    // move again to global position

    // max and min values for bouding box
    if (x > m_xMax) m_xMax = x;
    else if (x < m_xMin) m_xMin = x;
    if (y > m_yMax) m_yMax = y;
    else if (y < m_yMin) m_yMin = y;
    if (z > m_zMax) m_zMax = z;
    else if (z < m_zMin) m_zMin = z;
    
  }
}

////////////////////////////////////// CircularCDW //////////////////////////////////////

bool FLB::CircularCDW::isNodeInside(double xSI, double ySI, double zSI, std::vector<FLB::Point>& points, uint32_t idxEndPointBase)
{
  // calculate distance squared
  double d2 = FLB::Math::distanceSquaredPointLine(xSI, ySI,  zSI, points, idxEndPointBase, m_Radius); 
  if (d2 <= (m_Radius * m_Radius + m_Epsilon)) return true;
  return false;
}

void FLB::CircularCDW::setValue(double value, int type)
{
  m_Radius = value;
}

double FLB::CircularCDW::getValue(int type)
{
  return m_Radius;
}

////////////////////////////////////// RectangularCDW //////////////////////////////////////

bool FLB::RectangularCDW::isNodeInside(double xSI, double ySI, double zSI, std::vector<FLB::Point>& points, uint32_t idxEndPointBase)
{
  double xySlope = (points[idxEndPointBase].y - points[idxEndPointBase - 1].y)/(points[idxEndPointBase].x - points[idxEndPointBase - 1].x);
  double yDown = points[idxEndPointBase - 1].y + (xSI - points[idxEndPointBase - 1].x) * xySlope;
  double yUp = yDown + height;

  if (m_Is3D)
  {

  }

  else if (ySI >= (yDown - m_Epsilon) && ySI <= (yDown + m_Epsilon)) return true;
  return false;
}

void FLB::RectangularCDW::setValue(double value, int type)
{
  if (type == 0) height = value;
  else width = value;
}

double FLB::RectangularCDW::getValue(int type)
{
  if (type == 0) return height;
  else return width;
}
