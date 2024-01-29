#include "font.h"
#include "API.h"
#include "Charset.h"
#include "TightAtlasPacker.h"
#include "ext/import-font.h"
#include "texture.h"
#include <cstdint>
#include <memory>


namespace FLB
{
  template<typename T, typename S, int N, msdf_atlas::GeneratorFunction<S, N> GenFunc>
  static std::unique_ptr<FLB::Texture2D> createAndCacheAtlas(const std::string& fontName, float fontSize, const std::vector<msdf_atlas::GlyphGeometry>& glyphs, const msdf_atlas::FontGeometry& fontGeometry, uint32_t width, uint32_t height, FLB::API api)
  {
      msdf_atlas::GeneratorAttributes attributes;
      attributes.config.overlapSupport = true;
      attributes.scanlinePass = true;

      msdf_atlas::ImmediateAtlasGenerator<S, N, GenFunc, msdf_atlas::BitmapAtlasStorage<T, N>> generator(width, height);
      generator.setAttributes(attributes);
      generator.setThreadCount(8);
      generator.generate(glyphs.data(), (int)glyphs.size());

      msdfgen::BitmapConstRef<T, N> bitmap = (msdfgen::BitmapConstRef<T, N>)generator.atlasStorage();

      std::unique_ptr<FLB::Texture2D> texture = Texture2D::create(api, width, height, FLB::ImageFormat::RGB8);
      texture->setData((void*)bitmap.pixels, bitmap.width * bitmap.height * 3);
      return texture;
  }
}

/////////////////////////////////////////////////// Font ///////////////////////////////////////////////////

FLB::Font::Font(const std::filesystem::path& fontPath, FLB::API api)
{
  msdfgen::FreetypeHandle* ft = msdfgen::initializeFreetype();
  if(!ft) throw std::runtime_error("Failed to load FreeType");

  std::string fileString = fontPath.string();
  msdfgen::FontHandle* font = msdfgen::loadFont(ft, fileString.c_str());
  if (!font) throw std::runtime_error("Failed to load font: " + fileString);

  struct CharsetRange
  {
    uint32_t Begin, End;
  };

  // From imgui_draw.cpp
  static const CharsetRange charsetRanges[] =
  {
    { 0x0020, 0x00FF }
  };

  msdf_atlas::Charset charset;
  for (CharsetRange range : charsetRanges)
  {
    for (uint32_t c = range.Begin; c<= range.End; c++) charset.add(c);
  }

  double fontScale = 1.0;
  m_FontGeometry = std::make_unique<msdf_atlas::FontGeometry>(msdf_atlas::FontGeometry(&m_Glyphs));
  int glyphsLoaded = m_FontGeometry -> loadCharset(font, fontScale, charset);

  double emSize = 40.0;

  msdf_atlas::TightAtlasPacker atlasPacker;
  atlasPacker.setPixelRange(2.0);
  atlasPacker.setMiterLimit(1.0);
  atlasPacker.setPadding(0);
  atlasPacker.setScale(emSize);
  int reamining = atlasPacker.pack(m_Glyphs.data(), (int)m_Glyphs.size());
  if(reamining != 0) throw std::runtime_error("Failed to compute the array of glyphs");

  int width, height;
  atlasPacker.getDimensions(width, height);
  emSize = atlasPacker.getScale();

#define DEFAULT_ANGLE_THRESHOLD 3.0
#define LCG_MULTIPLIER 6364136223846793005ull
#define LCG_INCREMENT 1442695040888963407ull
#define THREAD_COUNT 8

  // if MSDF || MTSDF
  uint64_t coloringSeed = 0;
  bool expensiveColoring = false;
  if (expensiveColoring)
  {
    msdf_atlas::Workload([&glyphs = m_Glyphs, &coloringSeed](int i, int threadNo) -> bool {
    unsigned long long glyphSeed = (LCG_MULTIPLIER * (coloringSeed ^ i) + LCG_INCREMENT) * !!coloringSeed;
    glyphs[i].edgeColoring(msdfgen::edgeColoringInkTrap, DEFAULT_ANGLE_THRESHOLD, glyphSeed);
    return true;
    }, m_Glyphs.size()).finish(THREAD_COUNT);
  }
  else
  {
    unsigned long long glyphSeed = coloringSeed;
    for (msdf_atlas::GlyphGeometry& glyph : m_Glyphs)
    {
      glyphSeed *= LCG_MULTIPLIER;
      glyph.edgeColoring(msdfgen::edgeColoringInkTrap, DEFAULT_ANGLE_THRESHOLD, glyphSeed);
    }
  }

  m_AtlasTexture = FLB::createAndCacheAtlas<uint8_t, float, 3, msdf_atlas::msdfGenerator>("Test", (float)emSize, m_Glyphs, *m_FontGeometry, width, height, api);

  msdfgen::destroyFont(font);
  msdfgen::deinitializeFreetype(ft);
}

FLB::Font::~Font()
{
  
}
