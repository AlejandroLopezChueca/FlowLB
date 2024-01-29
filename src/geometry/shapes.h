#pragma once

#include "utils.h"
#include "io/csvReader.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>
#include <FL/Fl_Simple_Terminal.H>
#include <tuple>
#include <numbers>

namespace FLB 
{
  enum class TypeValue
  {
    Height = 0, xWidth, zWidth, x0, y0, z0, Radius, Thickness, xRotation, yRotation, zRotation
  };

  enum class TypeShape
  {
    Circle = 0, Rectangle, Imported2DShape
  };

  class Shape
  {
    public:
      virtual ~Shape() = default;
      Shape() = default;
      Shape(double x, double y, double z);

      virtual bool initShape(const std::filesystem::path& directoryPath, Fl_Simple_Terminal* terminal) {return true;};
      virtual bool isNodeInside(double xSI, double ySI, double zSI) {return 0;};

      virtual void setValue(const double value, TypeValue typeValue) {return;};
      virtual void setFilename(const std::string& nameFile) {return;};
      void setIs3D(bool is3D) { m_Is3D = is3D;}
      
      TypeShape typeShape;

    protected:

      double m_Epsilon = 1e-12;
      double m_x0 = 0.0, m_y0 = 0.0, m_z0 = 0.0; // Coordinates of the center (meters) in global system
      double m_xRotation = 0.0, m_yRotation = 0.0, m_zRotation = 0.0;// radians
      bool m_Is3D = false;
  };

  class CircleShape: public Shape
  {
    public:
      CircleShape() = default;
      CircleShape(double radius, double thickness, double x, double y, double z);

      bool isNodeInside(double xSI, double ySI, double zSI) override;

      void setValue(const double value, TypeValue typeValue) override;

    private:
      double m_Radius = 0.0f, m_Thickness = 0.0f; // (meters)
  };

  class RectangleShape: public Shape
  {
    public:
      RectangleShape() = default;

      /**
       * @brief Constructor
       *
       * @param[in]  height Height of the obstacle
       * @param[in]  xWidth
       * @param[in]  zWidth Only use in 3D analysis
       * @param[in]
       * @param[in]
       *
       *
       */
      RectangleShape(double height, double xWidth, double zWidth, double x, double y, double z);
      
      bool initShape(const std::filesystem::path& directoryPath, Fl_Simple_Terminal* terminal) override;

      bool isNodeInside(double xSI, double ySI, double zSI) override;
      
      void setValue(const double value, TypeValue typeValue) override;

    private:
      
      double m_Height = 0.0f, m_xWidth = 0.0f, m_zWidth = 0.0f; // meters
      double m_xMin = 0.0;
      double m_xMax = 0.0;
      double m_yMin = 0.0;
      double m_yMax = 0.0;
      double m_zMin = 0.0;
      double m_zMax = 0.0;
      
      std::vector<double> m_xCoordinates;
      std::vector<double> m_yCoordinates;
      std::vector<double> m_zCoordinates;

      // rows of rotation matrix
      double m_xRotationMatrix[3] = {0.0, 0.0, 0.0};
      double m_yRotationMatrix[3] = {0.0, 0.0, 0.0};
      double m_zRotationMatrix[3] = {0.0, 0.0, 0.0};

      bool m_RotatePoints = false;
  };

  class Imported2DShape: public Shape
  {
    public:
      Imported2DShape() = default;
      bool initShape(const std::filesystem::path& directoryPath, Fl_Simple_Terminal* terminal) override;

      bool isNodeInside(double xSI, double ySI, double zSI) override;

      void setValue(const double value, TypeValue typeValue) override;
      void setFilename(const std::string& nameFile) override {m_Filename = nameFile;};

    protected:
      std::pair<FLB::CSVReader, bool> createCSVReader(const std::filesystem::path& path, Fl_Simple_Terminal* terminal);
     
      /**
       * @ brief Calculation of the area according to the triangle formula. The only restriction that will be placed on the polygon for this technique to work is that the polygon must not be self intersecting. It also calculate the centroid in the same bucle 
       */
      void calculateAreaCentroid();
      /**
       *  @brief Recalculate the coordinates around its geometric center to rotate around it and move again to the global position indicated in the options
       */
      void moveRotateCoordinates();

      double m_xMin = 0.0;
      double m_xMax = 0.0;
      double m_yMin = 0.0;
      double m_yMax = 0.0;

      double m_xCentroid = 0.0;
      double m_yCentroid = 0.0;

      double m_Area = 0.0;

      std::vector<double> m_xCoordinates;
      std::vector<double> m_yCoordinates;

      std::string m_Filename;
  };
  
  class Imported3DShape: public Imported2DShape
  {
    public:
      Imported3DShape() = default;
      bool initShape(const std::filesystem::path& directoryPath, Fl_Simple_Terminal* terminal) override;

      bool isNodeInside(double xSI, double ySI, double zSI) override;

      void setValue(const double value, TypeValue typeValue) override;
      void setFilename(const std::string& nameFile) override {m_Filename = nameFile;}

    private:
      /**
       *  @brief Recalculate the coordinates around its geometric center to rotate around it and move again to the global position indicated in the options
       */
      void moveRotateCoordinates();

      double m_zMin = 0.0;
      double m_zMax = 0.0;

      double m_zCentroid = 0.0;

      std::vector<double> m_zCoordinates;

      std::string m_Filename;
  };

  enum class TypesCDW
  {
    CIRCLE = 0, RECTANGLE = 1
  };

  // Fordward declaration
  class Point;

  class CDW
  {
    public:
      virtual ~CDW() = default;

      /**
       * @brief It identifies if the node of the mesh is inside de CDW with the distance between the point and the line that connect the two points between which the node is located.
       * 
       * @param[in]
       * @param[in]
       *
       */
      virtual bool isNodeInside(double xSI, double ySI, double zSI, std::vector<FLB::Point>& points, uint32_t idxEndPointBase) {return 0;}
      virtual void setValue(double value, int type = 0) {return;}
      virtual double getValue(int type = 0) {return 0.0;}
      void setIs3D(bool is3D) { m_Is3D = is3D;}
      TypesCDW typeCDW;

    protected:
      double m_Epsilon = 1e-12;
      bool m_Is3D = false;

  };

  class CircularCDW: public CDW
  {
    public:
      //float radius;

      bool isNodeInside(double xSI, double ySI, double zSI, std::vector<FLB::Point>& points, uint32_t idxEndPointBase) override;
      void setValue(double value, int type = 0) override;
      double getValue(int type = 0) override;

      private:
      double m_Radius;
  };

  class RectangularCDW: public CDW
  {

    public:

      bool isNodeInside(double xSI, double ySI, double zSI, std::vector<FLB::Point>& points, uint32_t idxEndPointBase) override;
      void setValue(double value, int type) override;
      double getValue(int type) override;

      private:
	// height is in the y direction and width in the z direction
	// width is only used in a 3D case 
	double height, width;
      

  };
}
