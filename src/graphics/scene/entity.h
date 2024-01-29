#pragma once

//#include "components.h"
#include "graphics/UUID.h"
#include "scene.h"

#include "entt.hpp"
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <iostream>

namespace FLB
{
  class Scene;

  class Entity
  {
    public:
      Entity() = default;
      Entity(entt::entity entityHandle, FLB::Scene* scene);

      template<typename T, typename... Args>
      T& addComponent(Args&&... args)
      {
	if (hasComponent<T>())
	{
	  std::string message = "Entity already has component";
	  std::cout << message << "\n";
	  //ImGuiLogPanel::addLog(message, LogCategory::Warning);
	}
	T& component = m_Scene -> m_Registry.emplace<T>(m_EntityHandle, std::forward<Args>(args)...);
	return component;
      }

      template<typename T>
      T& getComponent() const
      {
	if (!hasComponent<T>())
	{
	  std::string message = "Entity does not hav component";
	  //ImGuiLogPanel::addLog(message, LogCategory::Error);
	}
	return m_Scene -> m_Registry.get<T>(m_EntityHandle);
      }

      template<typename T>
      void removeComponent()
      {
	if (!hasComponent<T>())
	{
	  std::string message = "Entity does not hav component";
	  //ImGuiLogPanel::addLog(message, LogCategory::Error);
	}
	else m_Scene -> m_Registry.remove<T>(m_EntityHandle);
      }

      template<typename T>
      bool hasComponent() const
      { 
	return m_Scene -> m_Registry.all_of<T>(m_EntityHandle);
      }

      FLB::UUID getUUID();
      std::string& getName();
      bool& getDrawAction();

      operator bool() const {return m_EntityHandle != entt::null;}
      operator entt::entity() const {return m_EntityHandle;}
      operator uint32_t() const {return (uint32_t)m_EntityHandle;}

      bool operator ==(const Entity& other) const
      {
	return m_EntityHandle == other.m_EntityHandle;// && m_Scene == other.m_Scene;
      }

      bool operator !=(const Entity& other) const
      {
	return !(*this == other);
      }

      //bool draw = true;

    private:
      entt::entity m_EntityHandle{entt::null};

      FLB::Scene* m_Scene = nullptr;


  };

}
