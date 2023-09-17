#type vertex
#version 330 core

layout(location = 0) in vec3 a_PoinstPosition;
layout(location = 1) in vec3 a_QuadPoints;
layout(location = 2) in vec2 a_QuadTextCoord;
layout(location = 3) in float a_Ux;

uniform mat4 u_ViewProjection;

out float velocity;
out vec2 textCoord;

void main()
{
  gl_Position = u_ViewProjection * vec4(a_QuadPoints, 1.0);
  velocity = a_Ux;
  textCoord = a_QuadTextCoord;
  
}

#type fragment
#version 450 core

layout(location = 0) out vec4 color;

//uniform sampler1D u_ColorMap;
 uniform sampler2D u_Texture;

//layout(binding = 0) uniform sampler1D u_ColorMap;
//layout(binding = 1) uniform sampler2D u_Texture;

in float velocity;
in vec2 textCoord;

void main()
{
  color = texture(u_Texture, textCoord) ;//+ vec4(velocity,velocity,0,1);
  //imageStore(u_Texture, textCoord, color_img);
  //color = texture(u_ColorMap, textCoord.x) ;
  //imageStore(u_Texture, textCoord, vec4(1));
  //imageStore(u_Texture, ivec2(400,400), vec4(5.0,5.0,5.0,1.0));
  //color = vec4(textCoord,0, 1.0);
}
