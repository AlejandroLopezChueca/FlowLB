#type vertex
#version 450 core

layout(location = 0) in vec3 a_QuadPoints;
layout(location = 1) in vec2 a_QuadTextCoord;

layout(std140, binding = 0) uniform u_ViewProjection
{
	mat4 viewProjection;
};

out vec2 textCoord;

void main()
{
  gl_Position = viewProjection * vec4(a_QuadPoints, 1.0);
  textCoord = a_QuadTextCoord;
}

#type fragment
#version 450 core

layout(location = 0) out vec4 color;

in vec2 textCoord;

uniform vec2 u_MaxMinValues;

layout(binding = 0) uniform sampler1D u_ColorMap;
layout(binding = 1) uniform sampler2D u_Texture;


void main()
{
  float velocity = (texture(u_Texture, textCoord).x - u_MaxMinValues.y)/(u_MaxMinValues.x - u_MaxMinValues.y);
  color = texture(u_ColorMap, max(velocity, 0));
}
