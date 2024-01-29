#type vertex
#version 450 core

layout(location = 0) in vec3 a_Pos;
layout(location = 1) in vec3 a_Offset;

uniform mat4 u_RecTransform;
uniform float u_Scale;
uniform vec4 u_MaxMinCoordDomain;

layout(std140, binding = 0) uniform u_ViewProjection
{
    mat4 viewProjection;
};

layout(binding = 1) uniform sampler2D u_Texture;

out float velocity;
out vec2 coordTexture;

void main()
{
    // Transform to [0,1]
    vec4 posOffset = u_RecTransform * vec4(a_Offset, 1.0f);
    coordTexture = vec2((posOffset.x - u_MaxMinCoordDomain.z)/(u_MaxMinCoordDomain.x - u_MaxMinCoordDomain.z), (posOffset.y - u_MaxMinCoordDomain.w)/(u_MaxMinCoordDomain.y - u_MaxMinCoordDomain.w));
    
    vec2 v = texture(u_Texture, coordTexture).xy;
    velocity = length(v);
    vec2 cosSen = v / max(1e-3, velocity);
    //vec2 cosSen = v * inversesqrt(max(1e-3, dot(v, v)));
    mat3 rot = mat3(cosSen.x, cosSen.y, 0.0f, -cosSen.y, cosSen.x, 0.0f, 0.0f, 0.0f, 1.0f);
    //mat2 rot = mat2(cosSen.x, cosSen.y, -cosSen.y, cosSen.x);
    gl_Position = viewProjection * (vec4(rot * a_Pos * u_Scale, 0.0f) +  posOffset);
}

#type fragment
#version 450 core

layout(location = 0) out vec4 color;

in float velocity;
in vec2 coordTexture;

layout(binding = 0) uniform sampler1D u_ColorMap;

void main()
{
    if (coordTexture.x < 0 || coordTexture.x > 1 || coordTexture.y < 0 || coordTexture.y > 1) discard;
    color = texture(u_ColorMap, max(velocity, 0));
}
