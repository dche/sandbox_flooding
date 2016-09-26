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

#define PARTICLE_PER_GRID   16

// SEE: https://graphics.stanford.edu/~seander/bithacks.html#InterleaveBMN
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

// SEE: http://cas.ee.ic.ac.uk/people/dt10/research/rngs-gpu-mwc64x.html
// NOTE: This PRGN has a short (2^63) period. The author suggests it is not
// suitable for using it in hours long simulation.
uint MWC64X(uint2 *state)
{
    enum { A=4294883355U };
    uint x=(*state).x, c=(*state).y;  // Unpack the state
    uint res=x^c;                     // Calculate the result
    uint hi=mul_hi(x,A);              // Step the RNG
    x=x*A+c;
    c=hi+(x<c);
    *state=(uint2)(x,c);              // Pack the state back up
    return res;                       // Return the next result
}

float uint2float(uint i) {
    i |= 0xFF800000;
    i &= 0x3FFFFFFF;
    return as_float(i) - 1.0f;
}

float random(uint2 *state) {
    uint rand = MWC64X(state);
    return uint2float(rand);
}

float2 random2(uint2 *state) {
    float a = random(state);
    float b = random(state);
    return (float2)(a, b);
}

// For computing `h`.
inline float poly6(float2 r, float l, float term_poly6) {
    // poly6 = (4 / (pi * l^8)) * (l^2 - r^2)^3
    // term_poly6 = (4 / (pi * l^8))
    return term_poly6 * pown(l * l - dot(r, r), 3);
}

// Re-calculates grid ID and raining.
__attribute__((reqd_work_group_size(PARTICLE_PER_GRID, 1, 1)))
__kernel void grid(
    // Unsorted particles.
    __global float8 *particles,
    __global const uint2 *start_index_map,
    int particle_count,
    __constant float *params,
    __global const float2 *clouds,
    int cloud_count,
    __global uint2 *random_state,
    float conversionRate,
    // 10 * 9 * PARTICLE_PER_GRID
    __local float2 *position_cache
) {
    // smoothing radius.
    float l = params[0];
    // sandbox size.
    float2 sandbox_size = (float2)(params[11], params[12]);

    // load all particles under the clouds.
    int lid = get_local_id(0);
    for (int ci = 0; ci < cloud_count; ci++) {
        // cloud position.
        float2 cp = clouds[ci];
        for (int i = 0; i < 9; i++) {
            float2 offset = (float2)(0.f, 0.f);
            // SEE: integrate.cl
            if (i == 6 || i == 1 || i == 8) offset.x = 1.f;
            if (i == 7 || i == 2 || i == 5) offset.x = -1.f;
            if (i == 7 || i == 4 || i == 6) offset.y = 1.f;
            if (i == 5 || i == 3 || i == 8) offset.y = -1.f;
            offset *= l;
            cp += offset;
            if (cp.x > 0.f && cp.y > 0.f && cp.x < sandbox_size.x && cp.y < sandbox_size.y) {
                uint grid = grid_id(cp, l);
                // start index
                int si = start_index_map[grid].x;
                // end index
                int ei = start_index_map[grid].y;
                if (ei < si) ei = particle_count;
                if (si + lid < ei) {
                    position_cache[ci * 9 * PARTICLE_PER_GRID + i * PARTICLE_PER_GRID + lid] =
                        particles[si + lid].s01;
                }
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int gid = get_global_id(0);
    // Particle: (position (3), grid_id, velocity, height of water body, bethmetry)
    float8 particle = particles[gid];
    uint2 rs = random_state[gid];
    // raining.
    // not virtual particle and there is cloud.
    if (particle.s2 >= 0.f && cloud_count > 0) {
        float n = random(&rs);
        // is it a rain drop?
        if (n < conversionRate) {
            // under which cloud?
            n = random(&rs) * convert_float(cloud_count);
            int ci = convert_int(n);
            // new position.
            float2 cloud = clouds[ci];
            // polar (r, theta)
            float2 v = random2(&rs);
            float r = l * v.x;
            float theta = v.y * M_PI_F * 2.f;
            float2 p = cloud + (float2)(cos(theta), sin(theta)) * r;
            // avoid collision.
            bool no_collision = true;
            int offset = ci * 9 * PARTICLE_PER_GRID;
            for (int i = 0; i < 9; i++) {
                for (int j = 0; j < PARTICLE_PER_GRID; j++) {
                    float2 cp = position_cache[offset + i * PARTICLE_PER_GRID + j];
                    // not valid particle (grid has particles less than `PARTICLE_PER_GRID`).
                    if (cp.x == 0.f || cp.y == 0.f) break;
                    // TODO: collision distance should be a parameter.
                    if (length(p - cp) < 2.f) {
                        no_collision = false;
                        break;
                    }
                }
                if (!no_collision) break;
            }
            if (no_collision) {
                // reset state.
                particle.s01 = p;
                particle.s2 = 1.f;
                // TODO: small initial velocity.
                particle.s45 = (float2)(0.f, 0.f);
            }
        }
        // store new random state back.
        random_state[gid] = rs;
    }
    // re-calculate grid id.
    particle.s3 = as_float(grid_id(particle.s01, l));
    particles[gid] = particle;
}
