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

import flat.math._
import flat.event.Mouse
import flat.io.cl.GLImage
import flat.io.gl.{ FragmentProgram, FrameBuffer, GLPlatform }
import flat.io.gl.sl._
import flat.io.gl.texture.{ RenderTexture, Sampler }
import flat.io.zeromq._
import flat.representation.graphic.{ Projector, Renderer }
import flat.util.{ Resource, TypedBuffer }
import flat.value.audio.analysis.Onset
import flat.value.color.Color
import flat.value.image._
import flat.value.image.filter._
import flat.world.{ Agent, World }

import org.bytedeco.javacpp.opencv_imgproc.medianBlur

import com.eleuth.sym.Scene

import scala.concurrent.ExecutionContext.Implicits.global

import com.jogamp.opengl.{ GL, GL2GL3, GL4 }

case class SandboxScene(
  sceneId: Int,
  title: String,
  description: String) extends Scene { outer =>

  defineControlEvent[Float](
    'contourInterval,
    "等高线间距",
    "等高线之间的距离。",
    50f,
    NumberDecoder(10f, 100f, "毫米"))

  defineControlEvent[Vector2](
    'lightDirection,
    "照明方向",
    "设置一个有向光源的水平方向。垂直方向总是为向下。",
    Vector2(1f, 1f),
    Vector2Decoder)

  defineControlEvent[Float](
    'cloudHeight,
    "云层的最低高度",
    "云层距Kinect镜头的最大距离。只有当手掌距Kinect镜头低于此值时，下雨效果才会出现。",
    1200f,
    NumberDecoder(1000, 1500))

  defineControlEvent[Float](
    'ground,
    "沙箱底距Kinect镜头的距离。",
    "沙箱底距Kinect镜头的距离",
    2.5f,
    NumberDecoder(2f, 2.5f, "米"))

  this.enable()

  override def useKinect = true

  private val cloudDetector = Agent()

  // depth map is filtered for increasing stability.
  private val filteredDepthMap = Image.withSize(GRAY16, 640, 480)

  // FIXME: should parameters be provided by command line arguments, or
  //        user controllable?
  // this filter runs on CPU.
  private val depthMapFilter = new SandboxScene.FrameFilter(640, 480, 1000, 2000, 9)

  // the fragment shader that draws to the screen.
  private var filter: Option[Filter] = None

  // distance between isolines, in mm.
  private def contourInterval = propertyValue[Float]('contourInterval).getOrElse(50f)

  private def lightDirection = propertyValue[Vector2]('lightDirection).getOrElse(Vector2(1, 1))

  private def cloudHeight = propertyValue[Float]('cloudHeight).getOrElse(1200f)

  private def ground = propertyValue[Float]('ground).getOrElse(2.5f)

  private lazy val clDevice = GLPlatform.clDevice.filter(_.isGLSharable)

  private val shallowWater =
    new ShallowWaterBehavior(
      32768,
      // depth map size in real scale.
      Vector2(2, 1.5f),
      // Distance from ground to Kinect camera, 2.5m by default.
      2.5f,
      // Sand box size in simulation scale.
      Vector2(640, 480),
      // Offset of the sand box relative to the Kinect depth map, in real
      // scale.
      Vector2.Zero,
      // The possibility that a particle should be converted to a new rain drop,
      // no matter if the particle is active or not.
      0.002f,
      // Radius of smoothing kernel, in simulation scale.
      10f,
      // Ratio between real and simulation scales.
      320f)

  /// FOR TESTING ONLY ///////////////////////////////////////////////////////

  // FBO for offline rendering simulated terrain. Used only in testing.
  private lazy val terrainFBO = new FrameBuffer()
  private lazy val terrainTex: Option[RenderTexture] = {
    val tex = new RenderTexture(640, 480, 0, GL2GL3.GL_R16)
    val fbo =
      GLPlatform.currentGL.flatMap { gl =>
        FrameBuffer.initFBO(gl, terrainFBO, tex, true)
      }
    if (!fbo.isDefined) {
      println("Failed in initialized terrain FBO.")
    }
    fbo.map(_ => tex)
  }

  private lazy val terrainShaderSrc =
    Resource.contentsOfTextFile(this, "terrainSim.glsl")

  private def terrainGen(width: Int, height: Int) =
    FragmentProgram(
      terrainShaderSrc,
      Seq(
        Uniform("time", outer.time),
        Uniform("resolution", Vector2(width, height))
      ))

  private def updateTerrain(gl: GL4): Unit = {
    terrainTex.map { tex =>
      val iv = new Array[Int](4)
      gl.glGetIntegerv(GL.GL_DRAW_FRAMEBUFFER_BINDING, iv, 0)
      val _currentFBO = iv(0)
      terrainFBO.bind(gl)
      gl.glGetIntegerv(GL.GL_VIEWPORT, iv, 0)
      gl.glViewport(0, 0, tex.width, tex.height)
      FilterKernel.execute(gl, terrainGen(tex.width, tex.height))
      gl.glViewport(iv(0), iv(1), iv(2), iv(3))
      gl.glBindFramebuffer(GL.GL_FRAMEBUFFER, _currentFBO)
    }
  }
  ////////////////////////////////////////////////////////////////////////////

  // FBO for offline rendering water texture through particle positions.
  // The texture is then used in the background shader to synthesize the
  // final image.
  private lazy val waterFBO = new FrameBuffer()
  private lazy val waterTex: Option[RenderTexture] = {
    val tex = new RenderTexture(640, 480, 0, GL2GL3.GL_R16)
    val fbo =
      GLPlatform.currentGL.flatMap { gl =>
        FrameBuffer.initFBO(gl, waterFBO, tex, true)
      }
    if (!fbo.isDefined) {
      println("Failed in initialized water texture FBO.")
    }
    fbo.map(_ => tex)
  }

  // for storing the intermediate result of blurring.
  private lazy val filterFBO = new FrameBuffer()
  private lazy val filterTex: Option[RenderTexture] = {
    val tex = new RenderTexture(640, 480, 0, GL2GL3.GL_R16)
    val fbo =
      GLPlatform.currentGL.flatMap { gl =>
        FrameBuffer.initFBO(gl, filterFBO, tex, true)
      }
    if (!fbo.isDefined) {
      println("Failed in initialized image filter FBO.")
    }
    fbo.map(_ => tex)
  }

  private lazy val blurFilters = GaussianBlur(5).filters

  private def updateWaterTex(gl: GL4): Unit = {
    for {
      wtex <- waterTex
      ftex <- filterTex
    } {
      val iv = new Array[Int](4)
      gl.glGetIntegerv(GL.GL_DRAW_FRAMEBUFFER_BINDING, iv, 0)
      val _currentFBO = iv(0)
      gl.glGetIntegerv(GL.GL_VIEWPORT, iv, 0)
      gl.glViewport(0, 0, wtex.width, wtex.height)
      shallowWater.smoothingParticles.map(_.render(gl, waterFBO))
      blurFilters.apply(gl, waterFBO, wtex, filterFBO, ftex, Sampler.texture)
      gl.glViewport(iv(0), iv(1), iv(2), iv(3))
      gl.glBindFramebuffer(GL.GL_FRAMEBUFFER, _currentFBO)
    }
  }

  private var clouds: List[Vector2] = Nil

  override def setup(
    driver: Agent,
    gr: Renderer,
    backgroundProjector: Projector): Unit = {

    if (!filter.isDefined) {
      for {
        shaderRsc <- Resource(this, "sandbox.glsl")
        shaderSrc <- shaderRsc.readString
      } {
        val f =
          new GeneratorKernel {
            def program: FragmentProgram = {
              val textures =
                (for {
                  // XXX: COMMENTED OUT FOR TESTING.
                  // strm <- depthMapStream
                  // img <- strm.propertyValue('frame).asInstanceOf[Option[Image[GRAY16.type]]]
                  // tex <- {
                  //   depthMapFilter.apply(img, filteredDepthMap)
                  //   filteredDepthMap.setNeedsUpdate()
                  //   filteredDepthMap.texture
                  // }
                  tex <- terrainTex
                  dev <- clDevice
                  gl <- GLPlatform.currentGL
                  wtex <- waterTex
                } yield {
                  for {
                    terrain <- GLImage(GL.GL_TEXTURE_2D, tex, dev)
                  } {
                    if (shallowWater.update(terrain, clouds)) {
                      updateWaterTex(gl)
                    } else {
                      updateTerrain(gl)
                    }
                  }
                  List(
                    UniformSampler2D("depthMap", tex, Sampler.screenSpace),
                    UniformSampler2D("waterTex", wtex, Sampler.texture))
                }).getOrElse(Nil)
              FragmentProgram(
                shaderSrc,
                Seq(
                  Uniform("time", time),
                  Uniform("resolution", Vector2(gr.width, gr.height)),
                  Uniform("contourInterval", contourInterval),
                  Uniform("lightDirection", lightDirection),
                  Uniform("ground", ground)
                ) ++ textures)
            }
          }
        filter = Some(f)
        backgroundProjector.pushFilter(f)
        // detect cloud.
        cloudDetector.on(Mouse('pressed), Mouse('x), Mouse('y)) { (self: Agent, prsd: Float, mx: Float, my: Float) =>
          if (prsd != 0f) {
            val x = mx * 640 / gr.width
            val y = my * 480 / gr.height
            clouds = List(Vector2(x, y))
          } else {
            clouds = Nil
          }
        }
      }
    } else {
      filter.map(backgroundProjector.pushFilter(_))
    }
  }
}

object SandboxScene {

  // NOTE: Algorithm borrowed from `SARndbox`.
  private class FrameFilter(
    width: Int,
    height: Int,
    minDistance: Int,
    maxDistance: Int,
    // Number of slots in each pixel's averaging buffer
    averageSlotCount: Int,
    // Maximum variance to consider a pixel stable
    maxVariance: Int = 4) {

    require(maxDistance > minDistance)
    require(minDistance > 0)
    require(maxVariance > 1)

    class StatsBuffer {

      val buffer = TypedBuffer.intBuffer(width * height * 3)

      def apply(k: Int): (Int, Int, Int) = {
        val i = k * 3
        (buffer(i), buffer(i + 1), buffer(i + 2))
      }

      def update(k: Int, stats: (Int, Int, Int)): Unit = {
        val i = k * 3
        buffer(i) = stats._1
        buffer(i + 1) = stats._2
        buffer(i + 2) = stats._3
      }
    }

    class AverageBuffer {

      var version = 0

      def advanceVersion(): Unit = {
        version = (version + 1) % averageSlotCount
      }

      val buffer = {
        val b = TypedBuffer.intBuffer(width * height * averageSlotCount)
        var i = 0
        while (i < b.length) {
          b(i) = 2048
          i += 1
        }
        b
      }

      def apply(k: Int): Int = {
        val i = k * averageSlotCount + version
        buffer(i)
      }

      def update(k: Int, depth: Int): Unit = {
        val i = k * averageSlotCount + version
        buffer(i) = depth
      }
    }

    // Minimum number of valid samples needed to consider a pixel stable
    val minSampleCount: Int = (averageSlotCount + 1) / 2
    // Amount by which a new filtered value has to differ from the current value to update
    val hysteresis = 3f

    val statsBuffer = new StatsBuffer()
    val averageBuffer = new AverageBuffer()
    val validFrame = Image.withSize(GRAY16, width, height)

    def apply(img: Image[GRAY16.type], outputFrame: Image[GRAY16.type]): Unit = {
      require(img.width == width && img.height == height)

      var j = 0
      while (j < height) {
        var i = 0
        while (i < width) {
          val k = i + j * width
          val d = img.data(k) & 0xFFFF

          // TODO: depth correction.

          val oldVal = averageBuffer(k)
          var (sampleCount, sum, variance) = statsBuffer(k)
          if (d >= minDistance && d <= maxDistance) { // XXX: use projected planes?
            // store new value
            averageBuffer(k) = d
            // update statistics
            sampleCount += 1
            sum += d
            variance += d * d
            if (oldVal != 2048) {
              sampleCount -= 1
              sum -= oldVal
              variance -= oldVal * oldVal
            }
            statsBuffer(k) = (sampleCount, sum, variance)
          }
          // output
          // is it stable?
          if (sampleCount > minSampleCount &&
            sampleCount * variance <= maxVariance * sampleCount * sampleCount + sum * sum) {
            // Check if the new depth-corrected running mean is outside the previous value's envelope:
            // float newFiltered = pdcPtr->correct(float(sPtr[1]) / float(sPtr[0]));
            // new depth is the running mean.
            val newVal = sum / sampleCount
            val validVal = validFrame.data(k) & 0xFFFF
            if (abs(newVal - validVal) > hysteresis) {
              validFrame.data(k) = newVal.toShort
            }
          }
          i += 1
        }
        j += 1
      }
      averageBuffer.advanceVersion()
      // CHECK: what if do blurring before averaging?
      medianBlur(validFrame, outputFrame, 5)
    }
  }
}
