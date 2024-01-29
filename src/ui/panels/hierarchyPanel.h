#pragma once

#include "graphics/API.h"
#include "graphics/texture.h"
#include "graphics/scene/entity.h"

#include <cstdint>
#include <array>
#include <memory>
#include <vector>

namespace FLB
{
  class HierarchyPanel
  {
    public:
      HierarchyPanel(FLB::API api);
      ~HierarchyPanel() = default;

      void setScene(FLB::Scene* scene);

      void createTextures(FLB::API api);

      void onImGuiRender(bool* open);

      const FLB::Entity& getSelectedEntity() const {return m_SelectedEntity;}

      static std::array<std::unique_ptr<FLB::Texture2D>, 4>& getColorMaps() {return s_ColorMaps;}

      private:
	void drawEntityNode(FLB::Entity& entity);
	void drawScalarVectorialFieldNode();

	void drawComponents();
	void drawScalarVectorialFieldProperties();

	FLB::Scene* m_Scene;

	FLB::Entity m_SelectedEntity;
	bool m_ScalarVectorialFieldSelected = false;

	static std::array<std::unique_ptr<FLB::Texture2D>, 4> s_ColorMaps;
	std::vector<std::string> m_AvailableFields = {"Velocity", "Pressure"};
	int m_CurrentIdxVectorRepresentation = 0;
	std::vector<std::string> m_NameVectorRepresentation = {"Magnitude", "X", "Y"};
	std::vector<std::string> m_NameScalarRepresentation = {"Magnitude"};
	
  };
}
