#type vertex
#version 450 core

layout(location = 0) in vec3 a_Pos;
layout(location = 1) in vec4 a_Color;
layout(location = 2) in mat4 a_Transform;

layout(std140, binding = 0) uniform u_ViewProjection
{
    mat4 viewProjection;
};

out vec4 lineColor;

void main()
{
    gl_Position = viewProjection * a_Transform * vec4(a_Pos, 1.0);
    lineColor = a_Color;
}

#type fragment
#version 450 core

layout(location = 0) out vec4 color;

in vec4 lineColor;

void main()
{
    color = lineColor;
}
