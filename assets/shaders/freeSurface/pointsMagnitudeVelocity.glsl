#type vertex
#version 450 core

layout(location = 0) in vec3 a_PoinstPosition;
layout(location = 1) in vec3 a_QuadPoints;
layout(location = 2) in vec2 a_QuadTextCoord;
layout(location = 3) in vec2 a_U;

layout(std140, binding = 0) uniform u_ViewProjection
{
	mat4 viewProjection;
};

out float velocity;

void main()
{
  gl_Position = viewProjection * vec4(a_PoinstPosition, 1.0);
  velocity = length(a_U);
}

#type fragment
#version 450 core

layout(location = 0) out vec4 color; 

in float velocity;

uniform vec2 u_MaxMinValues;

layout(binding = 0) uniform sampler1D u_ColorMap;

void main()
{
  float v = (velocity - u_MaxMinValues.y)/(u_MaxMinValues.x - u_MaxMinValues.y);
  color = texture(u_ColorMap, max(v, 0));
}
