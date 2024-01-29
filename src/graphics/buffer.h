#pragma once


#include "API.h"
//#include "rendererAPI.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <sys/stat.h>
#include <vector>
#include <string>
#include "FL/Fl_Simple_Terminal.H"

namespace FLB
{
  enum class ShaderDataType
  {
    None = 0, Float, Float2, SOAFloat2, Float3, Float4, Mat4
  };

  static uint32_t ShaderDataTypeSize(ShaderDataType type);

  struct BufferElement
  {
    std::string name;
    ShaderDataType type;
    uint32_t size;
    uint32_t offset;
    bool normalized;
    bool instanced;
    size_t countStrideSOA; // used only when the vertex attributes follow a SOA scheme

    BufferElement(ShaderDataType type_ , const std::string name_, bool instanced_ = false, size_t countStrideSOA_ = 0, bool normalized_ = false);

    uint32_t getComponentCount() const;
  };

  class BufferLayout
  {
    public:

      BufferLayout() {}

      BufferLayout(const std::initializer_list<BufferElement>& elements_);
      inline uint32_t getStride() const {return m_Stride;}

      const std::vector<BufferElement>& getElements() const {return m_Elements;}

      std::vector<BufferElement>::iterator begin() {return m_Elements.begin();}
      std::vector<BufferElement>::iterator end() {return m_Elements.end();}
      std::vector<BufferElement>::const_iterator begin() const {return m_Elements.begin();}
      std::vector<BufferElement>::const_iterator end() const {return m_Elements.end();}

    private:
      void calculateOffsetsAndStride();

      std::vector<BufferElement> m_Elements;
      uint32_t m_Stride = 0;
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
      virtual void resize(void* vertices, size_t size) = 0;

      static std::unique_ptr<VertexBuffer> create(FLB::API api, Fl_Simple_Terminal* terminal, void* vertices, size_t size);
  };

  class IndexBuffer
  {
    public:
      virtual ~IndexBuffer() = default;
      virtual void bind() const = 0;
      virtual void unbind() const = 0;

      static std::unique_ptr<IndexBuffer> create(FLB::API api, Fl_Simple_Terminal* terminal, uint32_t* indices, uint32_t count);
  };

  class UniformBuffer
  {
    public:
      virtual ~UniformBuffer() = default;
      virtual void bind() const = 0;
      virtual void unbind() const = 0;
      virtual void setData(const void* data) const = 0;

      static std::unique_ptr<UniformBuffer> create(FLB::API api, Fl_Simple_Terminal* terminal, size_t size, uint32_t bindingPoint);
  };

}
