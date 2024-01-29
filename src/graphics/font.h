#pragma once


#include <algorithm>
#include <filesystem>
#include <memory>
#include <vector>

#include <msdf-atlas-gen.h>

#include "API.h"
#include "FontGeometry.h"
#include "GlyphGeometry.h"
#include "texture.h"

namespace FLB
{


  class Font
  {
    public:
      Font(const std::filesystem::path& fontPath, FLB::API api);
      ~Font();

      const msdf_atlas::FontGeometry* getFontGeometry() const {return m_FontGeometry.get();}
      const Texture2D* getAtlasTexture() const {return m_AtlasTexture.get();}

    private:
      std::vector<msdf_atlas::GlyphGeometry> m_Glyphs;
      std::unique_ptr<msdf_atlas::FontGeometry> m_FontGeometry;
      std::unique_ptr<Texture2D> m_AtlasTexture;
  };
}
