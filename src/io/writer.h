#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <variant>

#include "FL/Fl_Simple_Terminal.H"

#include "geometry/mesh.h"
//#include "ui/app.h"

namespace FLB
{
  enum class TypesDataVTU
  {
    UInt8, Int32, Float32, Float64  
  };


  //template<typename T>
  class FieldData
  {
    public:
      //FieldData(std::string name, std::string typeData, int numberComponents, size_t numberPoints);
      virtual void writeData(std::ofstream& ofs) {return ;};
  };
 
  template<typename T>
  class ScalarFieldData: public FieldData
  {
    public:
      ScalarFieldData(std::string name, std::string typeData, size_t numberPoints, const std::vector<T>& data);
      void writeData(std::ofstream& ofs);

    private:
      const std::vector<T>& dataField;
      std::string nameField;
      std::string dataType;
      size_t numPoints;
  };

  template<typename T>
  class Vector3DFieldData: public FieldData
  {
    public:
      Vector3DFieldData(std::string name, std::string typeData, size_t numberPoints, const std::vector<T>& data);
      void writeData(std::ofstream& ofs);

    private:
      const std::vector<T>& dataField;
      std::string nameField;
      std::string dataType;
      size_t numPoints;
  };

  /**
   * Writer to VTU format.
   */
  class VTUWriter
  {
    public:
      //VTUWriter();
      //~VTUWriter();
      void savePointData(std::string pathSave, Mesh* mesh, Fl_Simple_Terminal* terminal);

      template<typename T>
      void addField(std::string typeData, std::string nameField, std::vector<T>& data, size_t numberPoints, bool isScalar);

    private:
      void writePoints(std::ofstream& ofs, const std::vector<float>& points, size_t numberPoints);
      void writeHeader(std::ofstream& ofs, size_t numberPoints, size_t numberCells);
      void writeFooter(std::ofstream& ofs);
      void writeFields(std::ofstream& ofs);

      //std::vector<FieldData*> fieldsData;
      std::vector<std::unique_ptr<FieldData>> fieldsData;
      //std::vector<std::variant<uint8_t, float, double>> fieldsData;



  };
}
