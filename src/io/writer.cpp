#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "FL/Fl_Simple_Terminal.H"
//#include "fluid/template_panel.h"
#include "writer.h"


////////////////////////////////////// ScalarFieldData //////////////////////////////////////
template<typename T>
FLB::ScalarFieldData<T>::ScalarFieldData(std::string name, std::string typeData, size_t numberPoints, const std::vector<T>& data, const double constantToSI)
  : m_NameField(name), m_DataType(typeData), m_NumPoints(numberPoints), m_DataField(data), m_ConstantToSI(constantToSI)  {}

template<typename T>
void FLB::ScalarFieldData<T>::writeData(std::ofstream &ofs)
{
  ofs << "<DataArray type=\"" << m_DataType << "\" Name=\"" << m_NameField << "\" NumberOfComponents=\"1\" format=\"ascii\">\n";

  for (unsigned int idx = 0; idx < m_NumPoints; ++idx)
  {
    // use of + to convert uchar to number (case of the flags)
    ofs << +m_DataField[idx] * m_ConstantToSI;
    ofs << "\n";
  }
  ofs << "</DataArray>\n";
}

////////////////////////////////////// VectorFieldData //////////////////////////////////////

template<typename T>
FLB::VectorFieldData<T>::VectorFieldData(std::string name, std::string typeData, size_t numberPoints, const std::vector<T>& data, uint32_t numberComponents, const double constantToSI, const bool changeSignYComponent)
  : m_NameField(name), m_DataType(typeData), m_NumPoints(numberPoints), m_DataField(data),  m_NumberComponents(numberComponents), m_ConstantToSI(constantToSI), m_ChangeSignYComponent(changeSignYComponent) {}

template<typename T>
void FLB::VectorFieldData<T>::writeData(std::ofstream &ofs)
{
  ofs << "<DataArray type=\"" << m_DataType << "\" Name=\"" << m_NameField << "\" NumberOfComponents=\"" + std::to_string(m_NumberComponents) +"\" format=\"ascii\">\n";

  const double constantChangeYSign = m_ChangeSignYComponent ? -1.0 : 1.0;

  if (m_NumberComponents == 2)
  { 
    for (unsigned int idx = 0; idx < m_NumPoints; idx++)
    {
      ofs << m_DataField[idx] * m_ConstantToSI << " "; // x
      ofs << m_DataField[idx + m_NumPoints] * m_ConstantToSI * constantChangeYSign << "\n"; // y
    }
  }
  else if (m_NumberComponents == 3)
  {
    for (unsigned int idx = 0; idx < m_NumPoints; idx++)
    {
      ofs << m_DataField[idx] * m_ConstantToSI << " ";
      ofs << m_DataField[idx + m_NumPoints] * m_ConstantToSI * constantChangeYSign << " ";
      ofs << m_DataField[idx + 2 * m_NumPoints] * m_ConstantToSI << "\n";
    }

  }
  ofs << "</DataArray>\n";
}

////////////////////////////////////// Writer //////////////////////////////////////
FLB::Writer::Writer(const std::filesystem::path directoryPath)
  : m_DirectoryPath(directoryPath)
{
}

template<typename T>
void FLB::Writer::addField(std::string typeData, std::string nameField, const std::vector<T>& data, size_t numberPoints, bool isScalar, double constantToSI, uint32_t numberComponents, bool changeSignYComponent)
{
  if (isScalar)
  {
    std::unique_ptr<FLB::ScalarFieldData<T>> field(new FLB::ScalarFieldData<T>(nameField, typeData, numberPoints, data, constantToSI));
  m_FieldsData.push_back(std::move(field));
  }
  else
  {
    std::unique_ptr<FLB::VectorFieldData<T>> field(new FLB::VectorFieldData<T>(nameField, typeData, numberPoints, data, numberComponents, constantToSI, changeSignYComponent));
  m_FieldsData.push_back(std::move(field));
  }
}

std::string FLB::Writer::getFormat(FLB::FileFormat fileFormat)
{
  switch (fileFormat) 
  {
    case FLB::FileFormat::vti: return "ImageData";
    case FLB::FileFormat::vtu: return "UnstructuredGrid"; 
  }
}

std::string FLB::Writer::getExtension(FLB::FileFormat fileFormat)
{
  switch (fileFormat) 
  {
    case FLB::FileFormat::vti: return ".vti";
    case FLB::FileFormat::vtu: return ".vtu"; 
  }
}

void FLB::Writer::writeHeader(FLB::FileFormat fileFormat)
{
  std::string type = getFormat(fileFormat);
  m_Ofs << "<VTKFile type=\"" + type + "\" version=\"1.0\">\n";
}

void FLB::Writer::writeFooter(FLB::FileFormat fileFormat)
{
  std::string type = getFormat(fileFormat);
  m_Ofs << "</Piece>\n";
  m_Ofs << "</" + type + ">\n";
  m_Ofs << "</VTKFile>\n";
}

void FLB::Writer::writePointData()
{
  m_Ofs << "<PointData>\n";

  if (m_FieldsData.size() > 0)
  {
    for (std::unique_ptr<FLB::FieldData>& field : m_FieldsData) field -> writeData(m_Ofs);
  }

  m_Ofs << "</PointData>\n";
}
void FLB::Writer::writeCellData()
{
  m_Ofs << "<CellData>\n";
  m_Ofs << "</CellData>\n";
}

void FLB::Writer::setGridData(const std::string& nameFormat, const std::initializer_list<std::array<std::string, 2>> elements)
{
  m_Ofs << "<" + nameFormat << " ";
  for (const std::array<std::string, 2>& element : elements)
  {
    m_Ofs << element[0] << "=\"" << element[1] <<"\" ";

  }
  m_Ofs << ">\n";
}

void FLB::Writer::setPieceData(const std::initializer_list<std::array<std::string, 2>> elements)
{
  m_Ofs << "<Piece ";
  for (const std::array<std::string, 2>& element : elements)
  {
    m_Ofs << element[0] << "=\"" << element[1] <<"\" ";
  }
  m_Ofs << ">\n";
}

void FLB::Writer::createTimeSeriesFile(FLB::FileFormat fileFormat, Fl_Simple_Terminal* terminal, double timeInterval)
{
  std::string extension = getExtension(fileFormat);
  std::string name = "loaderFiles" + extension + ".series";
  std::filesystem::path path = m_DirectoryPath / name ;
  m_Ofs.open(path);
  if (!m_Ofs.good())
  {
    terminal -> printf("An error ocurred while creating the file %s\n", path.c_str());
    m_Ofs.close();
    return;
  }
  m_Ofs << "{\n" << "\"file-series-version\" : \"1.0\",\n" << "\"files\" : [\n";
  for (int i = 0; i < m_CountFiles; i++)
  {
    name = "result_" + std::to_string(i) + extension;
    m_Ofs << "{\"name\" : \"" + name + "\", \"time\" : " + std::to_string(i * timeInterval) + "},\n";
  }
  m_Ofs << "]\n" << "}";

  m_Ofs.close();
}

bool FLB::Writer::createDirectory()
{
  int i = 0;
  std::string name = "results_" + std::to_string(i); 
  std::filesystem::path path = m_DirectoryPath / name;
  while (std::filesystem::exists(path))
  {
    i += 1;
    name = "results_" + std::to_string(i); 
    path = m_DirectoryPath / name;
  }
  m_DirectoryPath /= name; 
  return std::filesystem::create_directory(m_DirectoryPath);
}

////////////////////////////////////// VTIWriter //////////////////////////////////////
FLB::VTIWriter::VTIWriter(const std::filesystem::path directorySave, bool isMesh)
  : Writer(directorySave) 
{
  if (!isMesh) createDirectory();
}


void FLB::VTIWriter::getDataMesh(FLB::Mesh *mesh)
{
  m_x2Extent = mesh -> getNx() - 1;
  m_y2Extent = mesh -> getNy() - 1;
  m_z2Extent = mesh -> getNz() - 1;
  m_xOrigin = mesh ->getxMin();
  m_yOrigin = mesh ->getyMin();
  m_zOrigin = mesh ->getzMin();
  m_dx = mesh -> getSizeInterval();
  m_dy = mesh -> getSizeInterval();
  m_dz = mesh -> getSizeInterval();
}

void FLB::VTIWriter::writeData(FLB::Mesh* mesh, Fl_Simple_Terminal* terminal, bool isMesh, double timeInterval)
{
  if (!initFile(mesh, terminal, isMesh)) return;
  writePointData();
  writeCellData();
  writeFooter(FLB::FileFormat::vti);
  m_Ofs.close();
  m_CountFiles += 1;
  if (!isMesh) createTimeSeriesFile(FLB::FileFormat::vti, terminal, timeInterval);
  m_FieldsData.clear();
}

bool FLB::VTIWriter::initFile(FLB::Mesh* mesh, Fl_Simple_Terminal* terminal, bool isMesh)
{
  std::string name; 
  if (isMesh) name = "mesh.vti";
  else name = "result_" + std::to_string(m_CountFiles) + ".vti";
  std::filesystem::path path = m_DirectoryPath / name;

  m_Ofs.open(path);
  if (!m_Ofs.good())
  {
    terminal -> printf("An error ocurred while creating the file %s\n", path.c_str());
    m_Ofs.close();
    return false;
  }
  writeHeader(FLB::FileFormat::vti);
  getDataMesh(mesh);
  std::string extent = std::to_string(m_x1Extent) + " " + std::to_string(m_x2Extent) +" " + std::to_string(m_y1Extent) + " " + std::to_string(m_y2Extent) + " " + std::to_string(m_z1Extent) + " " + std::to_string(m_z2Extent);
  std::string origin = std::to_string(m_xOrigin) + " " + std::to_string(m_yOrigin) + " " + std::to_string(m_zOrigin);
  std::string spacing = std::to_string(m_dx) + " " + std::to_string(m_dy) + " " + std::to_string(m_dz);
  setGridData("ImageData", {{"WholeExtent", extent}, {"Origin", origin}, {"Spacing", spacing}});
  setPieceData({{"Extent", extent}});
  return true;
} 

////////////////////////////////////// VTUWriter //////////////////////////////////////
FLB::VTUWriter::VTUWriter(const std::filesystem::path directorySave)
  : Writer(directorySave) {}

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
  writeHeader(ofs, mesh -> getNumberPointsMesh(), 0);
  writePoints(ofs, mesh -> getCoordinatesPoints(), mesh -> getNumberPointsMesh());
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

void FLB::VTUWriter::writeFields(std::ofstream& ofs)
{
  if (m_FieldsData.size() == 0) return;
  ofs << "<PointData>\n";
;
  for (std::unique_ptr<FLB::FieldData>& field : m_FieldsData) field -> writeData(ofs);

  ofs << "</PointData>\n";
}

// Explicit initialization of the classes ad functions because they are instantiated in a diferent compilation unit

template class FLB::ScalarFieldData<uint8_t>;
template class FLB::VectorFieldData<float>;
template class FLB::VectorFieldData<double>;

template void FLB::Writer::addField<uint8_t>(std::string typeData, std::string nameField, const std::vector<uint8_t>& data, size_t numberPoints, bool isScalar, double  constantToSI, uint32_t numberComponents, bool changeSignYComponent);
template void FLB::Writer::addField<float>(std::string typeData, std::string nameField, const std::vector<float>& data, size_t numberPoints, bool isScalar, double constantToSI, uint32_t numberComponents, bool changeSignYComponent);
template void FLB::Writer::addField<double>(std::string typeData, std::string nameField, const std::vector<double>& data, size_t numberPoints, bool isScalar, double constantToSI, uint32_t numberComponents, bool changeSignYComponent);
