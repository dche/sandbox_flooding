//
// Sandbox Flooding
//
// Copyright (c) 2016, Eleuth Ltd.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#version 400 core

uniform float time;
uniform vec2 resolution;
uniform float contourInterval;
uniform vec2 lightDirection;
uniform float ground;
uniform sampler2D depthMap;
uniform sampler2D waterTex;

out vec4 color;

vec3 blend(vec4 fg, vec3 bg) {
    return mix(bg, fg.rgb, fg.a);
}

vec3 hermite(vec3 a, vec3 b, float t) {
    return mix(a, b, smoothstep(0., 1., t));
}

// d: distance to kinect camera, in mm.
vec3 land(in float d) {
    float h = ground * 1000. - d;

    // 0%
    vec3 c0 = vec3(.4, .5, .25);
    // 40%
    vec3 c1 = vec3(.7, .8, .5);
    float h1 = 500. * 0.4;
    float d1 = h1;
    // 70%
    vec3 c2 = vec3(.7, .4, .3);
    float h2 = 500. * 0.7;
    float d2 = h2 - h1;
    // 90%
    vec3 c3 = vec3(.85, .85, .85);
    float h3 = 500. * 0.9;
    float d3 = h3 - h2;

    if (h < h1) return hermite(c0, c1, h / d1);
    if (h < h2) return hermite(c1, c2, (h - h1) / d2);
    if (h < h3) return hermite(c2, c3, (h - h2) / d3);
    return hermite(c3, vec3(1.), (h - h3) / d3);
}

vec2 delta = vec2(.001,0);

float depth(in vec2 uv) {
    // depth in mm.
    return texture(depthMap, uv).r * 65535.0;
}

vec2 gradient(in vec2 uv) {
    float a = depth(uv + delta);
    float b = depth(uv - delta);
    float c = depth(uv + delta.yx);
    float d = depth(uv - delta.yx);
    return vec2(a - b, c - d);
}

float isoline(in vec2 uv, float h, float ref, float pas, float tickness) {
    vec2 grad = gradient(uv);
    float v = (0.21*resolution.x)*abs(mod(h-ref+pas*.5, pas)-pas*.5)/(length(grad/(2.*delta.x))*tickness);
    return (tickness > 1.) ? smoothstep(.1,.9, v) : clamp(v -.2, 0., 1.);
}

vec3 hsl2rgb(in vec3 hsl) {
    float h = hsl.x * 6.;
    vec3 rgb = clamp( abs(mod(h+vec3(0.0,4.0,2.0),6.0)-3.0)-1.0, 0.0, 1.0 );
    return hsl.z + hsl.y * (rgb-0.5)*(1.0-abs(2.0*hsl.z-1.0));
}

vec4 water(in vec2 uv) {
    float r = texture(waterTex, uv).r;
    float l = 1. + clamp((r - .99) * 70., 0., 1.);
    vec3 hsl = vec3(0.6, .86, .62 * l);
    vec3 rgb = hsl2rgb(hsl);
    return vec4(rgb, r * 8);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord.xy / resolution.xy;

    float d = depth(uv);
    float f = isoline(uv, d, 0., 25., 0.42);

    vec3 col = land(d);
    col = mix(col * 0.62, col, f);
    col = blend(water(uv), col);
    fragColor = vec4(col, 1.);
}

void main() {
    vec4 fragColor = vec4(0.);
    mainImage(fragColor, gl_FragCoord.xy);
    color = fragColor;
}
