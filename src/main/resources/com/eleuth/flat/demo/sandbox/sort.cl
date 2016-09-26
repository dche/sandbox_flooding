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

/// OpenCL Radix Sorting /////////////////////////////////////////////////////

// Based on,
// Philippe HELLUY, A portable implementation of the radix sort algorithm in OpenCL, HAL 2011.

#define _BITS 5  // number of bits in the radix
#define _RADIX (1 << _BITS) //  radix  = 2^_BITS

__kernel void count(
    __global const float4 *particles,
    // the counting bucket.
    __global int *grid_bins,
    int pass,
    int n,
    __local int *local_bins
) {
    int lid = get_local_id(0);
    int gid = get_global_id(0);
    int grpid = get_group_id(0);
    int groupCount = get_num_groups(0);
    int itemCount = get_local_size(0);

    // initialize local bins to `0`.
    for (int i = 0; i < _RADIX; i++) {
        local_bins[i * itemCount + lid] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // range of keys that are analyzed by the work item
    int size = n / groupCount / itemCount;
    int start = gid * size;

    float4 p;
    uint key;
    int shortkey;
    for (int i = start; i < start + size; i++) {
        p = particles[i * 2];
        key = as_uint(p.w);
        shortkey = (key >> (pass * _BITS)) & (_RADIX - 1);
        local_bins[shortkey * itemCount + lid]++;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // copy from local to global storage.
    for (int i = 0; i < _RADIX; i++) {
        grid_bins[i * groupCount * itemCount + grpid * itemCount + lid] =
            local_bins[i * itemCount + lid];
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
}

__kernel void scan(
    __global int *count,
    __global int *sum,
    __local int *temp
) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int decale = 1;
    int n = get_local_size(0) * 2;
    int grpid = get_group_id(0);

    // load input into local memory
    // up sweep phase
    temp[lid * 2] = count[gid * 2];
    temp[lid * 2 + 1] = count[gid * 2 + 1];

    // parallel prefix sum (algorithm of Blelloch 1990)
    for (int d = n >> 1; d > 0; d >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < d) {
            int ai = decale * (2 * lid + 1) - 1;
            int bi = decale * (2 * lid + 2) - 1;
            temp[bi] += temp[ai];
        }
        decale *= 2;
    }

    // store the last element in the global sum vector
    // (maybe used in the next step for constructing the global scan)
    // clear the last element
    if (lid == 0) {
        sum[grpid] = temp[n - 1];
        temp[n - 1] = 0;
    }

    // down sweep phase
    for (int d = 1; d < n; d *= 2) {
        decale >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lid < d) {
            int ai = decale * (2 * lid + 1) - 1;
            int bi = decale * (2 * lid + 2) - 1;

            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // write results to device memory
    count[2 * gid] = temp[2 * lid];
    count[2 * gid + 1] = temp[ 2 * lid + 1];
    barrier(CLK_GLOBAL_MEM_FENCE);
}

__kernel void paste_sum(
    __global int *count,
    __global const int *sum
) {

  int gid = get_global_id(0);
  int grpid = get_group_id(0);

  int s = sum[grpid];

  count[2 * gid] += s;
  count[2 * gid + 1] += s;

  barrier(CLK_GLOBAL_MEM_FENCE);
}

__kernel void reorder(
    __global const float8 *particle_in,
    __global const int *grid_bins,
    __global float8 *particle_out,
    int pass,
    int n,
    __local int *local_bins
) {
    int lid = get_local_id(0);
    int gid = get_global_id(0);
    int grpid = get_group_id(0);
    int groupCount = get_num_groups(0);
    int itemCount = get_local_size(0);

    // copy start indices to local memory.
    for (int i = 0; i < _RADIX; i++) {
        local_bins[i * itemCount + lid] =
            grid_bins[i * groupCount * itemCount + grpid * itemCount + lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // range of keys that are analyzed by the work item
    int size = n / groupCount / itemCount;
    int start = gid * size;

    float8 p;
    uint key;
    int newpos, shortkey;
    for (int i = start; i < start + size; i++) {
        p = particle_in[i];
        key = as_uint(p.s3);
        shortkey = (key >> (pass * _BITS)) & (_RADIX - 1);
        newpos = local_bins[itemCount * shortkey + lid];
        // most EXPENSIVE operation.
        particle_out[newpos] = p;
        newpos++;
        local_bins[itemCount * shortkey + lid] = newpos;
    }
    // barrier(CLK_GLOBAL_MEM_FENCE);
}

__kernel void clear_start_index(
    __global uint *start_index_map
) {
    int gid = get_global_id(0);
    start_index_map[gid * 2] = 0;
    start_index_map[gid * 2 + 1] = 0;
}

__kernel void start_index(
    __global const float8 *particles,
    __global uint *start_index_map
) {
    int gid = get_global_id(0);
    if (gid == 0) return;

    float8 p = particles[gid];
    float8 pp = particles[gid - 1];
    int grid = as_int(p.s3);
    int pgrid = as_int(pp.s3);

    if (grid != pgrid) {
        start_index_map[grid * 2] = gid;
        start_index_map[pgrid * 2 + 1] = gid;
    }
}
