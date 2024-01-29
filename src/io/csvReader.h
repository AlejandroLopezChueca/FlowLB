#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <FL/Fl_Simple_Terminal.H>
#include <fstream>
#include <iterator>
#include <vector>
#include <iostream>

namespace FLB 
{

  class CSVReader
  {
    public:

      CSVReader(const std::filesystem::path& path);
      //CSVReader(const CSVReader& csvReader);
      CSVReader(CSVReader&& csvReader) = default;
      ~CSVReader();

      bool isOpen() const {return m_FileIn.is_open();}

      bool isEmpty();

      bool checkMinNumberLines(const uint32_t minNumberLines);

      const std::string& readLine();

    private:

      void countNonVoidLines();

      std::string m_Line;
      char m_Delimiter = ';';
      std::ifstream m_FileIn;
      uint32_t m_CountLines = 0;

    public:
      class RowIterator
      {
	public:
	  using difference_type = std::ptrdiff_t;
	  using value_type = std::vector<std::string>;
	  using pointer = const std::vector<std::string>*;
	  using reference = const std::vector<std::string>&;
	  using iterator_category = std::input_iterator_tag;
	  
	  RowIterator(CSVReader* reader, uint32_t numRows, bool end = false, char delimiter = ';');

	  RowIterator& operator++() {nextLine(); return *this;}

	  reference operator*() const {return m_Row;}
	  pointer operator->() const {return &m_Row;}

	  bool operator==(const RowIterator& otherRowIterator) const {return m_IsEnded == otherRowIterator.m_IsEnded;}

	  bool operator!=(const RowIterator& otherRowIterator) const {
return !(*this == otherRowIterator);} 

	  void nextLine();

	private:
	  char m_Delimiter;
	  value_type m_Row;
	  CSVReader* m_CSVReader;
	  bool m_IsEnded = true;
	  bool m_EndIteration = false;
	  uint32_t m_CurrentRow = 0;
	  uint32_t m_NumRows = 0;
      };

      RowIterator begin() {return RowIterator(this, m_CountLines);}
      RowIterator end() {return RowIterator(this, m_CountLines, true);}
  };


}
