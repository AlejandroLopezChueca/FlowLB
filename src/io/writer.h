#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>
#include <variant>

#include "FL/Fl_Simple_Terminal.H"

#include "geometry/mesh.h"
#include "io/reader.h"
//#include "ui/app.h"

namespace FLB
{
  enum class TypesDataVTK
  {
    UInt8 = 0, Int32, Float32, Float64  
  };

  enum class FileFormat 
  {
    vti = 0, vtu
  };


  //template<typename T>
  class FieldData
  {
    public:
      //FieldData(std::string name, std::string typeData, size_t numberPoints, const std::vector<typename Tp>);
      virtual void writeData(std::ofstream& ofs) = 0; //{return ;};
  
  };
 
  template<typename T>
  class ScalarFieldData: public FieldData
  {
    public:
      ScalarFieldData(std::string name, std::string typeData, size_t numberPoints, const std::vector<T>& data, const double constantToSI);
      void writeData(std::ofstream& ofs) override;

    private:
      const std::vector<T>& m_DataField;
      std::string m_NameField;
      std::string m_DataType;
      size_t m_NumPoints;
      const double m_ConstantToSI;
  };

  template<typename T>
  class VectorFieldData: public FieldData
  {
    public:
      VectorFieldData(std::string name, std::string typeData, size_t numberPoints, const std::vector<T>& data, uint32_t numberComponents, const double constantToSI, const bool changeSignYComponent);
      void writeData(std::ofstream& ofs) override;

    private:
      const std::vector<T>& m_DataField;
      std::string m_NameField;
      std::string m_DataType;
      size_t m_NumPoints;
      uint32_t m_NumberComponents;
      const double m_ConstantToSI;
      const bool m_ChangeSignYComponent;
  };

  /*
   *
   *
   */
  class Writer
  {
    public:
      Writer(const std::filesystem::path directoryPath);
      
      template<typename T>
      void addField(std::string typeData, std::string nameField, const std::vector<T>& data, size_t numberPoints, bool isScalar, double constantToSI = 1.0, uint32_t numberComponents = 2, bool changeSignYComponent = false);
      
    protected:
      std::string getFormat(FLB::FileFormat fileformat);
      std::string getExtension(FLB::FileFormat fileformat);
      void writeHeader(FLB::FileFormat fileFormat);
      void writeFooter(FLB::FileFormat fileFormat);
      void writePointData();
      void writeCellData();
      void setGridData(const std::string& nameFormat, const std::initializer_list<std::array<std::string, 2>> elements);
      void setPieceData(const std::initializer_list<std::array<std::string, 2>> elements);

      void createTimeSeriesFile(FLB::FileFormat fileFormat, Fl_Simple_Terminal* terminal, double timeInterval);
      bool createDirectory();
      
      std::vector<std::unique_ptr<FieldData>> m_FieldsData;
      uint32_t m_CountFiles = 0;
      std::filesystem::path m_DirectoryPath;
      std::ofstream m_Ofs;
  };

  /*
   *
   *
   */
  class VTIWriter: public Writer
  {
    public:
      VTIWriter(const std::filesystem::path directorySave, bool isMesh = false);

      void getDataMesh(FLB::Mesh* mesh);
      void writeData(FLB::Mesh* mesh, Fl_Simple_Terminal* terminal, bool isMesh = false, double timeInterval = 0.0);

    private:
      bool initFile(FLB::Mesh* mesh, Fl_Simple_Terminal* terminal, bool isMesh);

      uint32_t m_x1Extent = 0, m_x2Extent = 0;
      uint32_t m_y1Extent = 0, m_y2Extent = 0;
      uint32_t m_z1Extent = 0, m_z2Extent = 0;

      double m_xOrigin = 0.0, m_yOrigin = 0.0, m_zOrigin = 0.0;
      double m_dx = 0.0, m_dy = 0.0, m_dz = 0.0;

  };

  /**
   * Writer to VTU format.
   */
  class VTUWriter: public Writer
  {
    public:
      VTUWriter(const std::filesystem::path directorySave);
      //VTUWriter();
      //~VTUWriter();
      void savePointData(std::string pathSave, Mesh* mesh, Fl_Simple_Terminal* terminal);

    private:
      void writePoints(std::ofstream& ofs, const std::vector<float>& points, size_t numberPoints);
      void writeHeader(std::ofstream& ofs, size_t numberPoints, size_t numberCells);
      void writeFooter(std::ofstream& ofs);
      void writeFields(std::ofstream& ofs);

      //std::vector<FieldData*> fieldsData;
      //std::vector<std::variant<uint8_t, float, double>> fieldsData;
      //
  };
}
