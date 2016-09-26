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

#define EPSILON         0.000001f
#define G               9.80665f
#define MAX_NEIGHBORS   32

__constant sampler_t
sampler = CLK_NORMALIZED_COORDS_TRUE |
          CLK_ADDRESS_CLAMP_TO_EDGE |
          CLK_FILTER_LINEAR;

float d2bathymetry(float d, float cam_d) {
    // convert to mm.
    d *= 65535.f;
    // convert to m.
    d *= 0.001f;
    // convert to distance to ground.
    float h = cam_d - d;
    // convert to simulation scale, where 1m == 320m;
    h *= 320.f;
    return h;
}

// Samples depth map.
__kernel void sample(
    __global const float8 *particle_gl,
    __global float8 *particle_cl,
    __read_only image2d_t depthMap,
    // SEE: ShallowWaterBehavior.scala.
    __constant float *params
) {

    int gid = get_global_id(0);
    float8 p = particle_gl[gid];

    float scale = params[7];
    float2 depth_map_size = (float2)(params[9], params[10]);
    float cam_d = params[13];
    float2 sandbox_offset = (float2)(params[14], params[15]);

    // IMPORTANT: only translation is concerned. Rotation/Scale/Shear is not
    // supported. i.e., output of projector must be strictly aligned with the
    // sand box.
    float2 pos = p.s01;
    // inactive particle.
    if (pos.x < EPSILON && pos.y < EPSILON) return;
    // convert to real scale, because `depth_map_size` and `sandbox_offset` use real scale.
    pos /= scale;
    pos += sandbox_offset;
    float2 uv = pos / depth_map_size;
    // sample gradient.
    // XXX: delta should match the size of particles.
    float2 delta = (float2)(0.002f, 0.f);
    float a = d2bathymetry(read_imagef(depthMap, sampler, uv + delta).x, cam_d);
    float b = d2bathymetry(read_imagef(depthMap, sampler, uv - delta).x, cam_d);
    float c = d2bathymetry(read_imagef(depthMap, sampler, uv + delta.yx).x, cam_d);
    float d = d2bathymetry(read_imagef(depthMap, sampler, uv - delta.yx).x, cam_d);
    float2 g = (float2)(a - b, c - d);
    // simulation scale of delta.x * 2.
    float2 ds = delta.x * scale * depth_map_size * 2.f;
    g /= ds;
    p.s67 = g;
    particle_cl[gid] = p;
}

uint z_curve_order(uint2 lp) {
    uint a = lp.x;
    uint b = lp.y;

    a = (a | a << 8) & 0x00FF00FF;
    a = (a | a << 4) & 0x0F0F0F0F;
    a = (a | a << 2) & 0x33333333;
    a = (a | a << 1) & 0x55555555;

    b = (b | b << 8) & 0x00FF00FF;
    b = (b | b << 4) & 0x0F0F0F0F;
    b = (b | b << 2) & 0x33333333;
    b = (b | b << 1) & 0x55555555;

    return a | (b << 1);
}

uint grid_id(const float2 pos, const float grid_size) {
    uint2 lp;
    lp.x = (uint)(pos.x / grid_size);
    lp.y = (uint)(pos.y / grid_size);
    return z_curve_order(lp);
}

// For computing `h`.
inline float poly6(float2 r, float l, float term_poly6) {
    // poly6 = (4 / (pi * l^8)) * (l^2 - r^2)^3
    // term_poly6 = (4 / (pi * l^8))
    return term_poly6 * pown(l * l - dot(r, r), 3);
}

// For computing grad(h) (pressure force term).
inline float2 spiky_gradient(float2 r, float2 u, float l, float term_spiky_gradient) {
    // spiky = (10 / (pi * l^5)) * (l - r)^3
    // spiky_gradient = (10 / (pi * l^5)) * 3 * (l - r)^2 * -1
    // term_spiky_gradient = -30 / (pi * l^5)
    float m = length(r);
    if (m < EPSILON) {
        return u * 0.5f * 0.00019f;
    }
    return term_spiky_gradient * (r / m) * pown(l - m, 2);
}

inline float2 viscosity_laplacian(float2 r, float l, float term_viscosity_laplacian) {
    // SEE: Sol11b, 3.2,
    // viscosity_laplacian = 40 / (pi * l ^ 5) * (l - r)
    // term_viscosity_laplacian = 40 / (pi * l^5)
    return term_viscosity_laplacian * (l - r);
}

#define PARTICLE_PER_GRID   16

__attribute__((reqd_work_group_size(32, 1, 1)))
__kernel void integrate
(
    __global const float8 *particle_cl,
    __global float8 *particle_gl,
    __global const uint2 *start_index_map,
    int particle_count,
    int grid_count,
    float dt,
    // SEE: ShallowWaterBehavior.scala
    __constant float *params,
    // number of slots = PARTICLE_PER_GRID * 9.
    __local float8 *particle_cache
) {
    // Each workgroup processes 1 grid.
    int grpid = get_group_id(0);
    if (grpid >= grid_count) return;
    // grid info (start index, end index).
    uint2 grid = start_index_map[grpid];
    // index of first particle of the grid.
    int si = grid.x;
    // no particles at all.
    if (si == 0) return;
    // end index.
    int ei = grid.y;
    // last grid that has particles.
    if (ei == 0) ei = particle_count;
    // number of particles in this grid.
    int nparticle = ei - si;

    int lid = get_local_id(0);
    int grpsz = get_local_size(0);  // 32.

    // smoothing radius.
    float l = params[0];
    float max_speed = params[1];
    float friction_const = params[2];
    // pre-computed integration terms.
    float term_poly6 = params[3];
    float term_sg = params[4];
    float term_sl = params[5];
    // unit volume of a particle.
    float volume = params[6];
    float virtual_volume = volume * 62.f;
    // viscosity factor of the liquid.
    float viscosity_const = params[8];
    // sandbox size.
    float2 sandbox_size = (float2)(params[11], params[12]);

    // the first particle, for computing neighbor grids.
    float8 particle = particle_cl[si];
    float2 position = particle.s01;

    // load neighbor particles into the local cache.
    if (lid < PARTICLE_PER_GRID) {
        for (int i = 0; i < 9; i++) {
            float2 offset = (float2)(0.f, 0.f);

            // sampling order.
            // 7 | 4 | 6
            // --+---+--
            // 2 | 0 | 1
            // --+---+--
            // 5 | 3 | 8
            if (i == 6 || i == 1 || i == 8) offset.x = 1.f;
            if (i == 7 || i == 2 || i == 5) offset.x = -1.f;
            if (i == 7 || i == 4 || i == 6) offset.y = 1.f;
            if (i == 5 || i == 3 || i == 8) offset.y = -1.f;

            offset *= l;
            float2 p = position + offset;
            if (p.x > 0.f && p.y > 0.f && p.x < sandbox_size.x && p.y < sandbox_size.y) {
                uint grid = grid_id(p, l);
                // start index
                int lsi = start_index_map[grid].x;
                // end index
                int lei = start_index_map[grid].y;
                if (lei < lsi) lei = particle_count;
                if (lsi + lid < lei) {
                    particle_cache[i * PARTICLE_PER_GRID + lid] = particle_cl[lsi + lid];
                }
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int pid = lid;
    while (pid < nparticle) {
        int rid = pid + si;
        pid += grpsz;

        particle = particle_cl[rid];
        position = particle.s01;
        // inactive particle.
        if (position.x < EPSILON && position.y < EPSILON) continue;
        // virtual particle.
        if (particle.s2 < 0) continue;

        // SWE:
        // dh/dt = -h * div(u) **Not used**.
        // h = volume
        // grad(h) = grad(volume)
        // Del(u) = -g * (grad(h) + grad(H)) + viscosity * Laplacian(u) + a_ext

        float h = 0.f;
        float2 grad_h = (float2)(0.f, 0.f);
        float2 lap_u = (float2)(0.f, 0.f);

        // how many neighbors found.
        int count = 0;
        int j = 0;
        // SPH.
        for (j = 0; j < 9 * PARTICLE_PER_GRID; j++) {
            if (count == MAX_NEIGHBORS) break;
            int k = j;
            // sample current grid first, and then sample neighbor grids evenly.
            if (j >= PARTICLE_PER_GRID) {
                int nj = j - PARTICLE_PER_GRID;
                k = (nj % 8) * PARTICLE_PER_GRID + nj / 8 + PARTICLE_PER_GRID;
            }
            float8 p = particle_cache[k];
            // empty slot.
            if (as_int(p.s3) == 0) continue;

            float2 pos = p.s01;
            float2 r = position - pos;
            if (length(r) < l) {
                float p6 = poly6(r, l, term_poly6);
                float2 sg = spiky_gradient(r, particle.s45 + p.s45, l, term_sg);
                float2 vl = viscosity_laplacian(r, l, term_sl);
                float V = volume;
                // virtual particle (solid body or boundary).
                if (p.s2 < 0.f) V = virtual_volume;
                if (p.s2 > EPSILON) lap_u += V * p.s45 * vl / p.s2;
                h += p6 * V;
                grad_h += sg * V;
                count++;
            }
        }

        // compute forces.
        float2 pressure = -G * grad_h;
        float2 terrain = -G * particle.s67;
        float2 viscosity = viscosity_const * lap_u;
        float2 force = terrain + pressure + viscosity;

        // integration
        float2 du_dt = force;
        // velocity.
        float2 du = du_dt * dt;
        float2 u = particle.s45 * friction_const + du;
        float m = length(u);
        if (m > max_speed) {
            u = u * max_speed / m;
        }
        particle.s45 = u;
        // height (to the bottom of the pool).
        particle.s2 = h;
        // position.
        float2 dp = particle.s45 * dt;
        position += dp;
        // further prevent particle from going outside of box.
        position = clamp(position, 1.1f, sandbox_size - 1.1f);
        particle.s01 = position;
        // assign to global buffer.
        particle_gl[rid] = particle;
    }
}
