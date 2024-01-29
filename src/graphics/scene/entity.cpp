#include "entity.h"
#include "components.h"

#include <cstdint>

FLB::Entity::Entity(entt::entity handle, FLB::Scene* scene)
   : m_EntityHandle(handle), m_Scene(scene)
{
}


FLB::UUID FLB::Entity::getUUID()
{
  //return getComponent<FLB::IDComponent>().ID;
}

std::string& FLB::Entity::getName()
{
  return getComponent<FLB::TagComponent>().tag;
}

bool& FLB::Entity::getDrawAction()
{
  return getComponent<FLB::DrawComponent>().draw;
}

