#type vertex
#version 450 core

layout(location = 0) in vec2 a_QuadPoints;
layout(location = 1) in vec2 a_QuadTextCoord;

uniform vec4 u_Color;

layout(std140, binding = 0) uniform u_ViewProjection
{
	mat4 viewProjection;
};

out vec2 textCoord;
out vec4 isoColor;

void main()
{
  gl_Position = viewProjection * vec4(a_QuadPoints, 0.1f, 1.0f);
  textCoord = a_QuadTextCoord;
  isoColor = u_Color;
}

#type fragment
#version 450 core

layout(location = 0) out vec4 color;

in vec2 textCoord;
in vec4 isoColor;

layout(binding = 0) uniform sampler2D u_Texture;

void main()
{
  if (texture(u_Texture, textCoord).x < 0.1f) discard;//color = vec4(0,0,0,1); //discard;
  else color = isoColor;
  //color = vec4(texture(u_Texture, textCoord).x, 0.0f, 0.0f, 1.0f);
  //color = texture(u_Texture, textCoord) + isoColor;
  //color = texture(u_Texture, textCoord);
}
