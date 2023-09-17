#pragma once

#include <cstdint>
#include <memory>
#include <sys/stat.h>
#include <vector>

#include "FL/Fl_Simple_Terminal.H"
#include "API.h"
//#include "rendererAPI.h"

namespace FLB
{
  enum class ShaderDataType
  {
    None = 0, Float, Float2, Float3
  };

  static uint32_t ShaderDataTypeSyze(ShaderDataType type);

  struct BufferElement
  {
    std::string name;
    ShaderDataType type;
    uint32_t size;
    uint32_t offset;
    bool normalized;

    BufferElement(ShaderDataType type_ ,const std::string name_, bool normalized_ = false);

    uint32_t getComponentCount() const;
  };

  class BufferLayout
  {
    public:

      BufferLayout() {}

      BufferLayout(const std::initializer_list<BufferElement>& elements_);
      inline uint32_t getStride() const {return stride;}

      const std::vector<BufferElement>& getElements() const {return elements;}

      std::vector<BufferElement>::iterator begin() {return elements.begin();}
      std::vector<BufferElement>::iterator end() {return elements.end();}
      std::vector<BufferElement>::const_iterator begin() const {return elements.begin();}
      std::vector<BufferElement>::const_iterator end() const {return elements.end();}

    private:
      void calculateOffsetsAndStride();

      std::vector<BufferElement> elements;
      uint32_t stride = 0;
  };

  class VertexBuffer
  {
    public:
      virtual ~VertexBuffer() = default;

      virtual void bind() const = 0;
      virtual void unbind() const = 0;
      virtual const BufferLayout& getLayout() const = 0;
      virtual void setLayout(const BufferLayout& layout) = 0;
      virtual unsigned int getVertexBufferID() const = 0;

      static std::unique_ptr<VertexBuffer> create(FLB::API api, Fl_Simple_Terminal* terminal, float* vertices, int size);
  };

  class IndexBuffer
  {
    public:
      virtual ~IndexBuffer() = default;
      virtual void bind() const = 0;
      virtual void unbind() const = 0;

      static std::unique_ptr<IndexBuffer> create(FLB::API api, Fl_Simple_Terminal* terminal, uint32_t* indices, uint32_t count);
  };

}
