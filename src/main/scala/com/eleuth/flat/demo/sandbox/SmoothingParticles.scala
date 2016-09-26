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

package com.eleuth.flat.demo.sandbox

import flat.io.gl._
import flat.io.gl.sl.{ Uniform, UniformSampler2D }
import flat.io.gl.texture.Sampler
import flat.math._
import flat.representation.graphic._
import flat.value.Space3D
import flat.value.color.Color
import flat.value.geometry.Geometry
import flat.value.image._
import flat.value.material.Material

import com.jogamp.opengl.{ GL, GL2ES3, GL3, GL4 }

private case class SmoothingParticles(
  particleCount: Int,
  // sand box size in simulation scale.
  sandboxSize: Vector2,
  volume: Float,
  vao: VertexArray) {

  private val mvpMatrix: Matrix =
    Matrix.ortho(0, sandboxSize.x, sandboxSize.y, 0, 0, max(sandboxSize.x, sandboxSize.y))

  private val kernelSprite = {
    val img = Image.withSize(GRAY16, 128, 128)
    var j = 0
    while (j < 128) {
      var i = 0
      while (i < 128) {
        // poly6 = (4 / (pi * l^8)) * (l^2 - r^2)^3
        // l = 1.0
        // After normalized, we got (1.0 - r^2)^3
        val x = (i - 63.5f) / 64
        val y = (j - 63.5f) / 64
        val r2 = x * x + y * y
        val poly6 =
          if (r2 >= 1) 0.0
          else pow(1 - r2, 3)
        img(i, j) = Color.gray(poly6.toFloat)
        i += 1
      }
      j += 1
    }
    img.setNeedsUpdate()
    img
  }

  def render(gl: GL4, target: FrameBuffer): Unit = {
    for {
      tex <- kernelSprite.texture
    } {
      val vertProg =
        VertexProgram(
          SmoothingParticles.vertexShaderSrc,
          Seq(
            Uniform("mvpMatrix", mvpMatrix),
            Uniform("volume", volume)))
      val fragProg =
        FragmentProgram(
          SmoothingParticles.fragShaderSrc,
          Seq(
            UniformSampler2D("kernel", tex, Sampler.texture)))
      val progs = Seq(vertProg, fragProg)
      target.bind(gl)
      // clear.
      gl.glClearColor(0, 0, 0, 0)
      gl.glClear(GL.GL_COLOR_BUFFER_BIT)
      // draw.
      gl.glEnable(GL3.GL_PROGRAM_POINT_SIZE)
      // NOTE: we're executed in GLRenderer#drawSurfaces#postProcess, where
      //       GL_BLEND is disabled.
      gl.glEnable(GL.GL_BLEND)
      gl.glBlendFunc(GL.GL_ONE, GL.GL_ONE)
      PipelineProxy(progs).drawArrays(
        gl,
        GL.GL_POINTS,
        particleCount,
        vao,
        Nil)
      gl.glDisable(GL3.GL_PROGRAM_POINT_SIZE)
      gl.glDisable(GL.GL_BLEND)
      target.unbind(gl)
    }
  }
}

private object SmoothingParticles {

  val vertexShaderSrc = """
#version 410 core

layout(location = 0) in vec4 position;

uniform mat4 mvpMatrix;
uniform float volume;

out gl_PerVertex
{
    vec4 gl_Position;
    float gl_PointSize;
};

layout(location = 0) out float density;

#define EPSILON 0.00001f
#define PI      3.14159f

void main() {
  density = 0.;
  // only active particle is concerned.
  if (position.x > EPSILON && position.y > EPSILON) {
    // density. literally the distance from water plane to the bottom of pool.
    density = position.z;
    float r = sqrt(volume / (density * PI));
    gl_PointSize = max(8., r * 8. / density);
  }
  gl_Position = mvpMatrix * vec4(vec3(position.xy, 0.), 1.);
}"""

  val fragShaderSrc = """
#version 410 core

// density is the distance from particle to bathymetry.
layout(location = 0) in float density;

uniform sampler2D kernel;

out vec4 color;

void main() {
  if (density == 0.) discard;
  float f = texture(kernel, gl_PointCoord.xy).r;
  color = vec4(min(1., f * density / 32.));
}"""
}
