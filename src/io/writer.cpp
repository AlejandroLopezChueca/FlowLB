#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include "FL/Fl_Simple_Terminal.H"
//#include "fluid/template_panel.h"
#include "writer.h"


template<typename T>
FLB::ScalarFieldData<T>::ScalarFieldData(std::string name, std::string typeData, size_t numberPoints, const std::vector<T>& data): nameField(name), dataType(typeData), numPoints(numberPoints), dataField(data)  {}

template<typename T>
void FLB::ScalarFieldData<T>::writeData(std::ofstream &ofs)
{
  ofs << "<DataArray type=\"" << dataType << "\" Name=\"" << nameField << "\" NumberOfComponents=\"1\" format=\"ascii\">\n";

  for (unsigned int idx = 0; idx < numPoints; ++idx)
  {
    // use of + to convert uchar to number
    ofs << +dataField[idx];
    ofs << "\n";
  }
  ofs << "</DataArray>\n";
}

template<typename T>
FLB::Vector3DFieldData<T>::Vector3DFieldData(std::string name, std::string typeData, size_t numberPoints, const std::vector<T>& data): nameField(name), dataType(typeData), numPoints(numberPoints), dataField(data) {}

template<typename T>
void FLB::Vector3DFieldData<T>::writeData(std::ofstream &ofs)
{
  ofs << "<DataArray type=\"" << dataType << "\" Name=\"" << nameField << "\" NumberOfComponents=\"3\" format=\"ascii\">\n";

  for (unsigned int idx = 0; idx < 3*numPoints; idx += 3)
  {
    ofs << dataField[idx] << " ";
    ofs << dataField[idx + 1] << " ";
    ofs << dataField[idx + 2];
    ofs << "\n";
  }
  ofs << "</DataArray>\n";
}

void FLB::VTUWriter::savePointData(std::string pathSave, Mesh *mesh, Fl_Simple_Terminal* terminal)
{
  std::ofstream ofs;
  ofs.open(pathSave.c_str());
  if (!ofs.good())
  {
    terminal -> printf("An error ocurred while creating the file %s\n", pathSave.c_str());
    ofs.close();
    return;
  }
  writeHeader(ofs, mesh -> numPointsMesh, 0);
  writePoints(ofs, mesh -> coordinatesPoints, mesh -> numPointsMesh);
  writeFields(ofs);
  ofs << "<Cells>\n";

  // List vertex id's i=0, ..., n_vertices associated with each cell c
  ofs << "<DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
  ofs << "</DataArray>\n";
  ofs << "<DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\" RangeMin=\""<<1e+299 << "\" RangeMax=\"" << -1e+299 << "\">\n";
  ofs << "</DataArray>\n";
  ofs << "</Cells>\n";
  writeFooter(ofs);
  ofs.close();
}

void FLB::VTUWriter::writeHeader(std::ofstream& ofs, size_t numberPoints, size_t numberCells)
{
  ofs << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\">\n";
  ofs << "<UnstructuredGrid>\n";
  ofs << "<Piece NumberOfPoints=\"" << numberPoints << "\" NumberOfCells=\"" << numberCells << "\">\n"; 
}

void FLB::VTUWriter::writeFooter(std::ofstream& ofs)
{
  ofs << "</Piece>\n";
  ofs << "</UnstructuredGrid>\n";
  ofs << "</VTKFile>\n";
}

void FLB::VTUWriter::writePoints(std::ofstream& ofs, const std::vector<float>& points, size_t numberPoints)
{
  ofs <<"<Points>\n";
  ofs <<"<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
  // Always is a 3D point (x, y, z)
  for (unsigned int idx = 0; idx < 3*numberPoints; idx += 3)
  {
    ofs << points[idx] << " ";
    ofs << points[idx + 1] << " ";
    ofs << points[idx + 2];
    ofs << "\n";
  }
  ofs << "</DataArray>\n";
  ofs << "</Points>\n";
}

template<typename T>
void FLB::VTUWriter::addField(std::string typeData, std::string nameField, std::vector<T>& data, size_t numberPoints, bool isScalar)
{
  if (isScalar)
  {
    std::unique_ptr<FLB::ScalarFieldData<T>> field(new FLB::ScalarFieldData<T>(nameField, typeData, numberPoints, data));
  fieldsData.push_back(std::move(field));
  }
  else
  {
    std::unique_ptr<FLB::Vector3DFieldData<T>> field(new FLB::Vector3DFieldData<T>(nameField, typeData, numberPoints, data));
  fieldsData.push_back(std::move(field));
  }
}

void FLB::VTUWriter::writeFields(std::ofstream& ofs)
{
  if (fieldsData.size() == 0) return;
  ofs << "<PointData>\n";
;
  for (std::unique_ptr<FLB::FieldData>& field : fieldsData) field -> writeData(ofs);

  ofs << "</PointData>\n";
}

// Explicit initialization of the classes ad functions because they are instantiated in a diferent compilation unit

template class FLB::ScalarFieldData<uint8_t>;
template class FLB::Vector3DFieldData<float>;
template class FLB::Vector3DFieldData<double>;

template void FLB::VTUWriter::addField<uint8_t>(std::string typeData, std::string nameField, std::vector<uint8_t>& data, size_t numberPoints, bool isScalar);
template void FLB::VTUWriter::addField<float>(std::string typeData, std::string nameField, std::vector<float>& data, size_t numberPoints, bool isScalar);
template void FLB::VTUWriter::addField<double>(std::string typeData, std::string nameField, std::vector<double>& data, size_t numberPoints, bool isScalar);
