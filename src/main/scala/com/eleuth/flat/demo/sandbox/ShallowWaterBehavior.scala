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

import flat.io.cl.{ GLBuffer => CLGLBuffer, _ }
import flat.io.gl.{ BufferProxy, GLPlatform, VertexArray }
import flat.io.gl.sl._
import flat.math._
import flat.util.{ Log, Resource, Time, TypedBuffer }
import flat.world.World

import com.jogamp.opengl.GL4

final class ShallowWaterBehavior(
  maxParticleCount: Int,
  depthMapSize: Vector2,
  cameraDistance: Float,
  sandboxSize: Vector2,
  sandboxOffset: Vector2,
  conversionRate: Float,
  kernelRadius: Float,
  scale: Float) {

  // spare spaces for virtual particles.
  require(maxParticleCount > (sandboxSize.x + sandboxSize.y) * 2)

  // CL:

  private lazy val device: Option[Device] =
    GLPlatform.clDevice.filter(_.isGLSharable)

  private lazy val gridProgram =
    device.flatMap(Program(ShallowWaterBehavior.gridSrc, _))

  private lazy val sortProgram =
    device.flatMap(Program(ShallowWaterBehavior.radixSortSrc, _))

  private lazy val integrationProgram =
    device.flatMap(Program(ShallowWaterBehavior.integrationSrc, _))

  // Particle: (float8 (position(2), density, grid id, velocity(2), bathymetry gradient(2))).

  private lazy val _particle: Option[(VertexArray, CLGLBuffer)] = {
    val buf = TypedBuffer.floatBuffer(maxParticleCount * 8)

    // populate virtual particles on boundaries.
    // NOTE: the radius of a particle is always `1`.
    val w = sandboxSize.x
    val h = sandboxSize.y
    var i = 0
    var j = 1
    // top, bottom
    while (j < w) {
      buf(i * 8) = j
      buf(i * 8 + 8) = j
      buf(i * 8 + 1) = h - 1
      buf(i * 8 + 9) = 1
      // flag to indicate this is a virtual particle.
      // SEE: sort.cl#grid() & integration.cl.
      buf(i * 8 + 2) = -1
      buf(i * 8 + 10) = -1
      i += 2
      j += 2
    }
    // left, right
    j = 3
    while (j < h - 1) {
      buf(i * 8) = w - 1
      buf(i * 8 + 8) = 1
      buf(i * 8 + 1) = j
      buf(i * 8 + 9) = j
      buf(i * 8 + 2) = -1
      buf(i * 8 + 10) = -1
      i += 2
      j += 2
    }

    val stride = 4 * 8
    val buffer = BufferProxy(buf)
    val pos_attrib = VertexAttributeVec4("position", buffer, stride, 0)
    for {
      dev <- device
      vao <- {
        GLPlatform.createVertexArray { gl: GL4 =>
          val va = (new VertexArray).create(gl).bind(gl)
          pos_attrib.glVertexAttrib(gl, 0)
          gl.glEnableVertexAttribArray(0)
          va.unbind(gl)
        }
      }
      glbuf <- buffer.buffer
      clglbuf <- CLGLBuffer(glbuf, dev)
    } yield {
      (vao, clglbuf)
    }
  }

  private def particleVao: Option[VertexArray] =
    _particle.map(_._1)

  private def particle_gl: Option[CLGLBuffer] =
    _particle.map(_._2)

  // For storing the intermediate results.
  private lazy val particle_cl =
    device.map(CLBuffer(maxParticleCount * 8 * 4, _))

  // For raining. 2 uint32
  private lazy val randomState =
    device.map { dev =>
      val buf = CLBuffer(maxParticleCount * 2 * 4, dev)
      var i = 0
      var ary = new Array[Int](2)
      while (i < maxParticleCount) {
        ary(0) = World.random.nextInt()
        ary(1) = World.random.nextInt()
        buf.write(CLint2, i, ary)
        i += 1
      }
      buf
    }

  // For counting and re-ordering.
  // Each work item has its own full size bins.
  private lazy val gridBins = {
    device.map(CLBuffer(binCount * 4, _))
  }

  // auxiliary bins.
  private lazy val gridSum =
    device.map(CLBuffer(binSplit * 4, _))

  // temporary vector when the sum is not needed
  private lazy val tempSum =
    device.map(CLBuffer(binSplit * 4, _))

  private lazy val gridCount = {
    def zOrder(x: Int, y: Int): Int = {
      var a = x
      var b = y
      a = (a | a << 8) & 0x00FF00FF
      a = (a | a << 4) & 0x0F0F0F0F
      a = (a | a << 2) & 0x33333333
      a = (a | a << 1) & 0x55555555

      b = (b | b << 8) & 0x00FF00FF
      b = (b | b << 4) & 0x0F0F0F0F
      b = (b | b << 2) & 0x33333333
      b = (b | b << 1) & 0x55555555

      return a | (b << 1);
    }
    zOrder(
      (sandboxSize.x / kernelRadius).toInt - 1,
      (sandboxSize.y / kernelRadius).toInt - 1) + 1
  }

  // storing the index of first particle and number of particles of a grid.
  private lazy val start_index_map = {
    Log.info(this, s"kernel radius: [$kernelRadius], grid: [$gridCount]")
    device.map(CLBuffer(gridCount * 8, _))
  }

  // Cloud.
  // Support at most `10` clouds.
  // Each cloud is represented by a `Vector2`, which is the coordinates in
  // simulation space. So the height and size of clouds are not used in raining.
  // All clouds have same size (20m, 20m).
  private lazy val cloudBuffer =
    device.map(CLBuffer(2 * 4 * 10, _))

  // PARAMETERS:

  private val workgroupCount = 16

  private val localWorkSize = 16

  private def globalWorkSize = localWorkSize * workgroupCount

  assume(maxParticleCount % globalWorkSize == 0)

  // IMPORTANT: must match with the value of `__BITS` in `sort.cl`.
  private val keySize = 5

  private val radix = pow(2, keySize).toInt

  private def binCount = globalWorkSize * radix

  // number of splits of the histogram.
  private val binSplit = 512

  // max grid number is half a million.
  private val gridIndexSize = 20

  // IMPORTANT: in current design, `passCount` must be a even number.
  private def passCount = gridIndexSize / keySize

  assume(passCount % 2 == 0)

  // 50ms
  private val maxTimeStep = 0.05f

  private var lastIntegrationTime = Time.now

  // volume of particle. Assume the height of a singular particle is 1m.
  // i.e., volume * poly6(0, kernelRadius) = 1.
  // for `kernelRadius == 10`, `volume ~= 100`.
  private val particleVolume: Float = {
    PI * kernelRadius * kernelRadius / 4.0
  }

  private lazy val integrationParams: Option[CLBuffer] =
    device.map { dev =>
      val buf = CLBuffer(16 * 4, dev)
      // pre-computed smoothing kernel terms.
      // term_poly6 = (4 / (pi * l^8))
      val term_poly6: Float = 4.0 / (PI * pow(kernelRadius, 8.0))
      // term_spiky_gradient = -30 / (pi * l^5)
      val term_spiky_gradient: Float = -30.0 / (PI * pow(kernelRadius, 5.0))
      // term_spiky_laplacian = 40 / (pi * l^5)
      val term_spiky_laplacian: Float = 40.0 / (PI * pow(kernelRadius, 5.0))
      // of water.
      val viscosity_const = 0.00089f
      // the speed of a free falling body falls after 50m.
      val maxSpeed = 31.3f
      //
      val friction_const = 0.9962f;

      buf.write(CLfloat, 0, kernelRadius)
      buf.write(CLfloat, 1, maxSpeed)
      buf.write(CLfloat, 2, friction_const)
      buf.write(CLfloat, 3, term_poly6)
      buf.write(CLfloat, 4, term_spiky_gradient)
      buf.write(CLfloat, 5, term_spiky_laplacian)
      buf.write(CLfloat, 6, particleVolume)
      buf.write(CLfloat, 7, scale)
      buf.write(CLfloat, 8, viscosity_const)
      buf.write(CLfloat, 9, depthMapSize.x)
      buf.write(CLfloat, 10, depthMapSize.y)
      buf.write(CLfloat, 11, sandboxSize.x)
      buf.write(CLfloat, 12, sandboxSize.y)
      buf.write(CLfloat, 13, cameraDistance)
      buf.write(CLfloat, 14, sandboxOffset.x)
      buf.write(CLfloat, 15, sandboxOffset.y)
      buf
    }

  private[sandbox] lazy val smoothingParticles: Option[SmoothingParticles] =
    particleVao.map { vao =>
      new SmoothingParticles(maxParticleCount, Vector2(640, 480), particleVolume, vao)
    }

  // Phase 0: Sorting,
  //       1: Sample, Integration,
  private var phase = 1

  private def grid(clouds: List[Vector2]): Unit = {
    for {
      dev <- device
      prog <- gridProgram
      ptcls <- particle_gl
      simap <- start_index_map
      params <- integrationParams
      cloudsBuf <- cloudBuffer
      rs <- randomState
    } {
      // fill `cloudBuffer`.
      val cloudCount = clouds.length min 10
      val ary = new Array[Float](2)
      clouds.take(cloudCount).zipWithIndex.foreach {
        case (p, i) =>
          ary(0) = p.x
          ary(1) = p.y
          cloudsBuf.write(CLfloat2, i, ary)
      }
      prog.call(
        "grid",
        (maxParticleCount, 16, 0),
        ptcls,
        simap,
        maxParticleCount,
        params,
        cloudsBuf,
        cloudCount,
        rs,
        conversionRate,
        // 16 = PARTICLE_PER_GRID
        LocalArgument(10 * 9 * 16 * 8))
    }
  }

  private def sort(): Unit = {

    def count(pass: Int): Unit = {
      val particle: Option[KernelArgument] =
        if (pass % 2 == 0) particle_gl.map(GLBufferArgument(_))
        else particle_cl.map(BufferArgument(_))
      for {
        dev <- device
        prog <- sortProgram
        ptcls <- particle
        grid_bins <- gridBins
      } {
        prog.call(
          "count",
          (globalWorkSize, localWorkSize, 0),
          ptcls,
          grid_bins,
          pass,
          maxParticleCount,
          LocalArgument(localWorkSize * radix * 4))
      }
    }

    def scan(): Unit = {
      for {
        dev <- device
        prog <- sortProgram
        grid_bins <- gridBins
        grid_sum <- gridSum
        temp_sum <- tempSum
      } {
        val gws = binCount / 2
        val lws = gws / binSplit
        val cacheSz = binSplit max (binCount / binSplit)
        prog.call(
          "scan",
          (gws, lws, 0),
          grid_bins,
          grid_sum,
          LocalArgument(cacheSz))
        prog.call(
          "scan",
          Tuple1(binSplit / 2),
          grid_sum,
          temp_sum,
          LocalArgument(binSplit))
        prog.call(
          "paste_sum",
          (gws, lws, 0),
          grid_bins, grid_sum)
      }
    }

    def reorder(pass: Int): Unit = {
      val (particle_in, particle_out) =
        if (pass % 2 == 0) (particle_gl.map(GLBufferArgument(_)), particle_cl.map(BufferArgument(_)))
        else (particle_cl.map(BufferArgument(_)), particle_gl.map(GLBufferArgument(_)))
      for {
        dev <- device
        prog <- sortProgram
        ptcls_in <- particle_in
        grid_bins <- gridBins
        ptcls_out <- particle_out
      } {
        prog.call(
          "reorder",
          (globalWorkSize, localWorkSize, 0),
          ptcls_in,
          grid_bins,
          ptcls_out,
          pass,
          maxParticleCount,
          LocalArgument(localWorkSize * radix * 4))
      }
    }

    def start_index(): Unit = {
      for {
        dev <- device
        prog <- sortProgram
        ptcls <- particle_gl
        simap <- start_index_map
      } {
        prog.call(
          "clear_start_index",
          Tuple1(gridCount),
          simap)
        prog.call(
          "start_index",
          Tuple1(maxParticleCount),
          ptcls,
          simap)
      }
    }

    var pass = 0
    while (pass < passCount) {
      count(pass)
      scan()
      reorder(pass)
      pass += 1
    }
    start_index()
  }

  private def sample(terrain: GLImage): Unit = {
    for {
      dev <- device
      prog <- integrationProgram
      ptcls_gl <- particle_gl
      ptcls_cl <- particle_cl
      params <- integrationParams
    } {
      prog.call(
        "sample",
        Tuple1(maxParticleCount),
        ptcls_gl,
        ptcls_cl,
        terrain,
        params)
    }
  }

  private def integrate(): Unit = {
    for {
      dev <- device
      prog <- integrationProgram
      ptcls_gl <- particle_gl
      ptcls_cl <- particle_cl
      simap <- start_index_map
      params <- integrationParams
    } {
      //
      val dt = maxTimeStep min (Time.now - lastIntegrationTime).seconds
      prog.call(
        "integrate",
        (gridCount * 32, 32, 0),
        ptcls_cl,
        ptcls_gl,
        simap,
        maxParticleCount,
        gridCount,
        dt,
        params,
        LocalArgument(16 * 9 * 32))
    }
  }

  def update(terrain: GLImage, clouds: List[Vector2]): Boolean = {
    if (phase % 2 == 0) {
      grid(clouds)
      sort()
    } else {
      sample(terrain)
      integrate()
      for {
        ptcls_cl <- particle_cl
        if false // phase % 200 == 1
      } {
        var i = 0
        while (i < maxParticleCount) {
          for {
            f8 <- ptcls_cl.read(CLfloat8, i)
            grid <- ptcls_cl.read(CLint, i * 8 + 3)
          } {
            val x = f8(0)
            val y = f8(1)
            val h = f8(2)
            if (x != 0 && y != 0 && h >= 0) {
              println(s"particle[$i]: (pos: [${x}, ${y}, ${h}], grid: [$grid], speed: [${f8(4)}, ${f8(5)}], del(H): ${f8(6)}, ${f8(7)})")
            }
          }
          i += 1
        }
      }
    }
    phase += 1
    phase % 2 == 1
  }
}

object ShallowWaterBehavior {

  private val gridSrc = Resource.contentsOfTextFile(this, "grid.cl")
  private val radixSortSrc = Resource.contentsOfTextFile(this, "sort.cl")
  private val integrationSrc = Resource.contentsOfTextFile(this, "integration.cl")
}
