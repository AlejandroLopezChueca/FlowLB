#include "csvReader.h"
#include <cstddef>
#include <string>
#include <algorithm>

FLB::CSVReader::CSVReader(const std::filesystem::path& path)
  : m_FileIn(path)
{
  countNonVoidLines();
}

FLB::CSVReader::~CSVReader()
{
  m_FileIn.close();
}

bool FLB::CSVReader::isEmpty()
{
  if (m_CountLines == 0) return true;
  else return false;
}

bool FLB::CSVReader::checkMinNumberLines(const uint32_t minNumberLines)
{
  if (m_CountLines < minNumberLines) return false;
  else return true;
}

const std::string& FLB::CSVReader::readLine()
{
  std::getline(m_FileIn, m_Line);
  // remove whitespaces
  m_Line.erase(std::remove(m_Line.begin(), m_Line.end(), ' '), m_Line.end());
  return m_Line;
}

void FLB::CSVReader::countNonVoidLines()
{
  std::string line;
  while (std::getline(m_FileIn, line))
  {
    if (!line.empty()) m_CountLines += 1;
  }

  //return stream to the beginning
  m_FileIn.clear();
  m_FileIn.seekg(0, std::ios::beg);  
}

/////////////////////////// RowIterator /////////////////////////////////

FLB::CSVReader::RowIterator::RowIterator(FLB::CSVReader* reader, uint32_t numRows, bool end, char delimiter)
  : m_CSVReader(reader), m_NumRows(numRows), m_Delimiter(delimiter)
{
  if (!end) 
  {
    m_Row.reserve(5);
    m_IsEnded = false;
    nextLine();
  }
}

void FLB::CSVReader::RowIterator::nextLine()
{
  if (m_EndIteration) m_IsEnded = true;
  std::string line = m_CSVReader -> readLine();
  if (line.empty()) return;
  
  m_Row.clear();

  size_t prevPos = 0;
  size_t pos = 0;
  while (pos != std::string::npos) 
  {
    pos = line.find_first_of(m_Delimiter, prevPos);
    m_Row.push_back(line.substr(prevPos, pos - prevPos));
    prevPos = pos + 1;
  }
  m_CurrentRow += 1;
  if (m_CurrentRow == m_NumRows) m_EndIteration = true;
}
