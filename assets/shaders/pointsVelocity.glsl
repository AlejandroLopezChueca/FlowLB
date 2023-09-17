#type vertex
#version 330 core

layout(location = 0) in vec3 a_PoinstPosition;
layout(location = 1) in vec3 a_QuadPoints;
layout(location = 2) in vec2 a_QuadTextCoord;
layout(location = 3) in float a_Ux;

uniform mat4 u_ViewProjection;

out float velocity;

void main()
{
  gl_Position = u_ViewProjection * vec4(a_PoinstPosition, 1.0);// vec4(a_Ux,0,0,1.0);
  velocity = a_Ux;
}

#type fragment
#version 330 core

layout(location = 0) out vec4 color; 

uniform sampler1D u_ColorMap;
in float velocity;

void main()
{
  color = texture(u_ColorMap, velocity);
}
