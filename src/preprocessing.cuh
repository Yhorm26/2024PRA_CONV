#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include "mykernelParamType.cuh"

__device__ __inline__ void init_mask(short& mask, int tile_idx, int tile_idy, int weight){
    mask = 0xffff;
    if (tile_idx == 0 && tile_idy == 0)     mask &= 0xeee0;
    if (tile_idy == 0 && tile_idx > 0 && tile_idx < weight - 1)   mask &= 0xfff0;
    if (tile_idy == 0 && tile_idx == weight - 1 )   mask &= 0x7770;
    if (tile_idx == 0 && tile_idy > 0 && tile_idy < weight - 1)    mask &= 0xeeee;
    if (tile_idx == weight - 1 && tile_idy > 0 && tile_idy < weight - 1)    mask &= 0x7777;
    if (tile_idy == weight - 1 && tile_idx == 0)     mask &= 0x0eee;
    if (tile_idy == weight - 1 && tile_idx > 0 && tile_idx < weight - 1)   mask &= 0x0fff;
    if (tile_idx == weight - 1 && tile_idy == weight - 1)    mask &= 0x0777;
}

__device__ __inline__ void prefetch_input(short& mask, int weight, _Float16 *input, _Float16 *output){
    short x, offset;

    if (mask == 0xffff) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            offset = (i - 1) * weight;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                x = (i << 2) + j;
                output[x] = input[offset + j - 1];
            }
        }
    }
    else {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            offset = (i - 1) * weight;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                x = (i << 2) + j;
                output[x] = 0.0f;
                if (mask & (1 << x)) {
                    output[x] = input[offset + j - 1];
                }
            }
        }
    }
}

__device__ __inline__ void calculate_transform_pin(_Float16 *tile, _Float16 *result){
    // *
    // *          BT matrix             input tile                B matrix
    // *  |   1,   0,  -1,   0|   |  x1,  x2,  x3,  x4|    |   1,   0,   0,  0|
    // *  |   0,   1,   1,   0|   |  x5,  x6,  x7,  x8|    |   0,   1,  -1,  1|
    // *  |   0,  -1,   1,   0|   |  x9, x10, x11, x12|    |  -1,   1,   1,  0|
    // *  |   0,   1,   0,  -1|   | x13, x14, x15, x16|    |   0,   0,   0, -1|
    // *
    result[0] = tile[0] - tile[2] - tile[8] + tile[10];
    result[1] = tile[1] + tile[2] - tile[9] - tile[10];
    result[2] = tile[2] - tile[1] + tile[9] - tile[10];
    result[3] = tile[1] - tile[3] - tile[9] + tile[11];
    result[4] = tile[4] - tile[6] + tile[8] - tile[10];
    result[5] = tile[5] + tile[6] + tile[9] + tile[10];
    result[6] = tile[6] - tile[5] - tile[9] + tile[10];
    result[7] = tile[5] - tile[7] + tile[9] - tile[11];
    result[8] = tile[6] - tile[4] + tile[8] - tile[10];
    result[9] = tile[9] - tile[5] - tile[6] + tile[10];
    result[10] = tile[5] - tile[6] - tile[9] + tile[10];
    result[11] = tile[7] - tile[5] + tile[9] - tile[11];
    result[12] = tile[4] - tile[6] - tile[12] + tile[14];
    result[13] = tile[5] + tile[6] - tile[13] - tile[14];
    result[14] = tile[6] - tile[5] + tile[13] - tile[14];
    result[15] = tile[5] - tile[7] - tile[13] + tile[15];
}


__global__  void GgGT(mykernelParamType param) {
    // *
    // *     G matrix              filter              GT matrix
    // *  |   1,   0,  0 |   |  x1,  x2,  x3|    |   1, 0.5, 0.5,  0|
    // *  | 0.5, 0.5, 0.5|   |  x4,  x5,  x6|    |   0, 0.5,-0.5,  0|
    // *  | 0.5,-0.5, 0.5|   |  x7,  x8,  x9|    |   0, 0.5, 0.5,  1|
    // *  |   0,   0,   1|
    // *
    int idx = (blockIdx.x + blockIdx.y * gridDim.x ) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    _Float16 *load_filter = param.pweight + idx * 9;

    _Float16 buffer[25]; // 3*3 + 4*4;
    _Float16 *g = buffer, *C = buffer + 9;

    reinterpret_cast<ushort3*>(g)[0] = reinterpret_cast<ushort3*>(load_filter)[0];
    reinterpret_cast<ushort3*>(g)[1] = reinterpret_cast<ushort3*>(load_filter)[1];
    reinterpret_cast<ushort3*>(g)[2] = reinterpret_cast<ushort3*>(load_filter)[2];

    C[0]  = g[0];
    C[1]  = 0.5f * (g[0] + g[1] + g[2]);
    C[2]  = 0.5f * (g[0] - g[1] + g[2]);
    C[3]  = g[2];
    C[4]  = 0.5f * (g[0] + g[3] + g[6]);
    C[5]  = 0.25f * (g[0] + g[1] + g[2] + g[3] + g[4] + g[5] + g[6] + g[7] + g[8]);
    C[6]  = 0.25f * (g[0] - g[1] + g[2] + g[3] - g[4] + g[5] + g[6] - g[7] + g[8]);
    C[7]  = 0.5f * (g[2] + g[5] + g[8]);
    C[8]  = 0.5f * (g[0] - g[3] + g[6]);
    C[9]  = 0.25f * (g[0] + g[1] + g[2] - g[3] - g[4] - g[5] + g[6] + g[7] + g[8]);
    C[10] = 0.25f * (g[0] - g[1] + g[2] - g[3] + g[4] - g[5] + g[6] - g[7] + g[8]);
    C[11] = 0.5f * (g[2] - g[5] + g[8]);
    C[12] = g[6];
    C[13] = 0.5f * (g[6] + g[7] + g[8]);
    C[14] = 0.5f * (g[6] - g[7] + g[8]);
    C[15] = g[8];
        
    int offset = param.k * param.c;
    int row = idx % param.c;
    int col = idx / param.c;
    #pragma unroll
    for (short i = 0; i < 16; i++) {
        param.transform_filter[row * param.k + col + i * offset] = C[i]; // 转置了一下
    }
}


__global__  void GgGT2(mykernelParamType param) {
    short bx = blockIdx.x, by = blockIdx.y;
    short tid = threadIdx.y * blockDim.x + threadIdx.x;
    short warp_id = tid / 64, lane_id = tid % 64;

    _Float16 *load_filter = param.pweight + (by * 16 + lane_id / 4) * param.c * 9 + (bx * 16 + warp_id * 4 + lane_id % 4) * 9;

    _Float16 input[9], result[16]; 

    reinterpret_cast<ushort3*>(input)[0] = reinterpret_cast<ushort3*>(load_filter)[0];
    reinterpret_cast<ushort3*>(input)[1] = reinterpret_cast<ushort3*>(load_filter)[1];
    reinterpret_cast<ushort3*>(input)[2] = reinterpret_cast<ushort3*>(load_filter)[2];

    result[0]  = input[0];
    result[1]  = 0.5f * (input[0] + input[1] + input[2]);
    result[2]  = 0.5f * (input[0] - input[1] + input[2]);
    result[3]  = input[2];
    result[4]  = 0.5f * (input[0] + input[3] + input[6]);
    result[5]  = 0.25f * (input[0] + input[1] + input[2] + input[3] + input[4] + input[5] + input[6] + input[7] + input[8]);
    result[6]  = 0.25f * (input[0] - input[1] + input[2] + input[3] - input[4] + input[5] + input[6] - input[7] + input[8]);
    result[7]  = 0.5f * (input[2] + input[5] + input[8]);
    result[8]  = 0.5f * (input[0] - input[3] + input[6]);
    result[9]  = 0.25f * (input[0] + input[1] + input[2] - input[3] - input[4] - input[5] + input[6] + input[7] + input[8]);
    result[10] = 0.25f * (input[0] - input[1] + input[2] - input[3] + input[4] - input[5] + input[6] - input[7] + input[8]);
    result[11] = 0.5f * (input[2] - input[5] + input[8]);
    result[12] = input[6];
    result[13] = 0.5f * (input[6] + input[7] + input[8]);
    result[14] = 0.5f * (input[6] - input[7] + input[8]);
    result[15] = input[8];
        
    int offset = param.k * param.c;
    _Float16 *load_result = param.transform_filter + bx * 16 * param.c + (by / 2) * 32 * 16 + (warp_id * 4 + lane_id % 4) * 32 + (by % 2) * 16 + lane_id / 4;
    #pragma unroll
    for (short i = 0; i < 16; i++) {
        load_result[0] = result[i]; // 转置了一下
        load_result += offset;
    }
}


__global__  void GgGT4(mykernelParamType param) {
    int idx = (blockIdx.x + blockIdx.y * gridDim.x ) * (blockDim.x * blockDim.y);
    short tid = (threadIdx.y * blockDim.x) + threadIdx.x;
    _Float16 *load_filter = param.pweight + idx * 12 * 9 + tid * 9;

    _Float16 input_mem[18], result[12][16]; 
    _Float16 *input = input_mem, *buffer = input_mem + 9;
    _Float16 *swap;

    reinterpret_cast<ushort3*>(input)[0] = reinterpret_cast<ushort3*>(load_filter)[0];
    reinterpret_cast<ushort3*>(input)[1] = reinterpret_cast<ushort3*>(load_filter)[1];
    reinterpret_cast<ushort3*>(input)[2] = reinterpret_cast<ushort3*>(load_filter)[2];

    #pragma unroll
    for(short iter = 0; iter < 12; iter ++){
        if(iter < 11){
            load_filter += 256 * 9;
            reinterpret_cast<ushort3*>(buffer)[0] = reinterpret_cast<ushort3*>(load_filter)[0];
            reinterpret_cast<ushort3*>(buffer)[1] = reinterpret_cast<ushort3*>(load_filter)[1];
            reinterpret_cast<ushort3*>(buffer)[2] = reinterpret_cast<ushort3*>(load_filter)[2];
        }

        result[iter][0]  = input[0];
        result[iter][1]  = 0.5f * (input[0] + input[1] + input[2]);
        result[iter][2]  = 0.5f * (input[0] - input[1] + input[2]);
        result[iter][3]  = input[2];
        result[iter][4]  = 0.5f * (input[0] + input[3] + input[6]);
        result[iter][5]  = 0.25f * (input[0] + input[1] + input[2] + input[3] + input[4] + input[5] + input[6] + input[7] + input[8]);
        result[iter][6]  = 0.25f * (input[0] - input[1] + input[2] + input[3] - input[4] + input[5] + input[6] - input[7] + input[8]);
        result[iter][7]  = 0.5f * (input[2] + input[5] + input[8]);
        result[iter][8]  = 0.5f * (input[0] - input[3] + input[6]);
        result[iter][9]  = 0.25f * (input[0] + input[1] + input[2] - input[3] - input[4] - input[5] + input[6] + input[7] + input[8]);
        result[iter][10] = 0.25f * (input[0] - input[1] + input[2] - input[3] + input[4] - input[5] + input[6] - input[7] + input[8]);
        result[iter][11] = 0.5f * (input[2] - input[5] + input[8]);
        result[iter][12] = input[6];
        result[iter][13] = 0.5f * (input[6] + input[7] + input[8]);
        result[iter][14] = 0.5f * (input[6] - input[7] + input[8]);
        result[iter][15] = input[8];

        swap = input, input = buffer, buffer = swap;
    }
        
    _Float16 *load_result = param.transform_filter + idx * 12 + tid;
    int offset = param.k * param.c;
    #pragma unroll
    for (short i = 0; i < 16; i++) {
        #pragma unroll
        for(short j = 0; j < 12; j++){
            load_result[j * 256] = result[j][i];     //没有转置
        }
        load_result += offset;
    }
}


__global__  void GgGT5(mykernelParamType param) {
    short bx = blockIdx.x, by = blockIdx.y;
    short tid = threadIdx.y * blockDim.x + threadIdx.x;
    short warp_id = tid / 64, lane_id = tid % 64;

    _Float16 *load_filter = param.pweight + by * 32 * param.c * 9 + bx * 64 * 9 + warp_id * 16 * 9 + (lane_id / 16) * param.c * 9 + (lane_id % 16) * 9;

    _Float16 input_mem[18], result[8][16]; 
    _Float16 *input = input_mem, *buffer = input_mem + 9;
    _Float16 *swap;

    reinterpret_cast<ushort3*>(input)[0] = reinterpret_cast<ushort3*>(load_filter)[0];
    reinterpret_cast<ushort3*>(input)[1] = reinterpret_cast<ushort3*>(load_filter)[1];
    reinterpret_cast<ushort3*>(input)[2] = reinterpret_cast<ushort3*>(load_filter)[2];

    #pragma unroll
    for(short iter = 0; iter < 8; iter++){
        if(iter < 7){
            load_filter += param.c * 4 * 9;
            reinterpret_cast<ushort3*>(buffer)[0] = reinterpret_cast<ushort3*>(load_filter)[0];
            reinterpret_cast<ushort3*>(buffer)[1] = reinterpret_cast<ushort3*>(load_filter)[1];
            reinterpret_cast<ushort3*>(buffer)[2] = reinterpret_cast<ushort3*>(load_filter)[2];
        }

        result[iter][0]  = input[0];
        result[iter][1]  = 0.5f * (input[0] + input[1] + input[2]);
        result[iter][2]  = 0.5f * (input[0] - input[1] + input[2]);
        result[iter][3]  = input[2];
        result[iter][4]  = 0.5f * (input[0] + input[3] + input[6]);
        result[iter][5]  = 0.25f * (input[0] + input[1] + input[2] + input[3] + input[4] + input[5] + input[6] + input[7] + input[8]);
        result[iter][6]  = 0.25f * (input[0] - input[1] + input[2] + input[3] - input[4] + input[5] + input[6] - input[7] + input[8]);
        result[iter][7]  = 0.5f * (input[2] + input[5] + input[8]);
        result[iter][8]  = 0.5f * (input[0] - input[3] + input[6]);
        result[iter][9]  = 0.25f * (input[0] + input[1] + input[2] - input[3] - input[4] - input[5] + input[6] + input[7] + input[8]);
        result[iter][10] = 0.25f * (input[0] - input[1] + input[2] - input[3] + input[4] - input[5] + input[6] - input[7] + input[8]);
        result[iter][11] = 0.5f * (input[2] - input[5] + input[8]);
        result[iter][12] = input[6];
        result[iter][13] = 0.5f * (input[6] + input[7] + input[8]);
        result[iter][14] = 0.5f * (input[6] - input[7] + input[8]);
        result[iter][15] = input[8];

        swap = input, input = buffer, buffer = swap;
    }
        
    int offset = param.k * param.c;

    _Float16 *load_result = param.transform_filter + (bx * 4 + warp_id) * param.c * 16 + by * 32 * 16 + (lane_id % 16) * 32 + lane_id / 16;
    #pragma unroll
    for (short i = 0; i < 16; i++) {
        #pragma unroll
        for(short j = 0; j < 8; j++){
            load_result[j * 4] = result[j][i];
        }
        load_result += offset;
    }
}


__global__ void BTdB1(mykernelParamType param) {
    short tx = threadIdx.x, ty = threadIdx.y;
    short bx = blockIdx.x, by = blockIdx.y;
    short tid = ty * blockDim.x + tx;

    _Float16 *input = param.pin + (by * param.c + bx) * param.w * param.w;

    __shared__ _Float16 share_mem[64 * 64];
    #pragma unroll
    for(short i = 0; i < 16; i++){
        share_mem[tid * 16 + i] = input[tid * 16 + i];
    }

    __syncthreads();

    _Float16 *share = share_mem + 2 * ty * param.w + 2 * tx;

    short mask = 0xffff;
    init_mask(mask, tx, ty, param.w / 2);

    _Float16 tile_mem[32], result[64];
    _Float16 *tile = tile_mem, *tile_buffer = tile_mem + 16, *swap;

    prefetch_input(mask, param.w, share, tile);

    #pragma unroll
    for(short iter = 0; iter < 4; iter++){
        if(iter < 3){
            share += 16 * param.w;
            init_mask(mask, tx, ty + 8 * (iter + 1), param.w / 2);
            prefetch_input(mask, param.w, share, tile_buffer);
        }
        calculate_transform_pin(tile, result + iter * 16);
        swap = tile, tile = tile_buffer, tile_buffer = swap;
    }

    int offset2 = param.c * param.h * param.w / 4;
    _Float16 *load_result = param.transform_pin + param.c * param.h * param.w * 4 * by;
    int row = (bx / 16) * (param.h * param.w / 128) * 16 + bx % 16 + (tid / 32) * 16;
    short col = tid % 32;
    #pragma unroll
    for (short i = 0; i < 16; i++) {
        #pragma unroll
        for(short j = 0; j < 4; j++){
            load_result[j * 256 * 16 + row * 32 + col] = result[j * 16 + i];
        }
        load_result += offset2;
    }
}


__global__ void BTdB2(mykernelParamType param) {
    short tx = threadIdx.x, ty = threadIdx.y;
    short bx = blockIdx.x, by = blockIdx.y;
    short tid = ty * blockDim.x + tx;

    _Float16 *input = param.pin + (by * param.c + bx * 4) * param.w * param.w;

    __shared__ _Float16 share_mem[32 * 32 * 4];


    reinterpret_cast<ulonglong4*>(share_mem)[tid] = reinterpret_cast<ulonglong4*>(input)[tid];

    __syncthreads();

    _Float16 *share = share_mem + 2 * ty * param.w + 2 * tx;

    short mask = 0xffff;
    init_mask(mask, tx, ty, param.w / 2);

    _Float16 tile_mem[32], result[64];
    _Float16 *tile = tile_mem, *tile_buffer = tile_mem + 16, *swap;

    prefetch_input(mask, param.w, share, tile);

    #pragma unroll
    for(short iter = 0; iter < 4; iter++){
        if(iter < 3){
            share += param.h * param.w;

            prefetch_input(mask, param.w, share, tile_buffer);
        }
        calculate_transform_pin(tile, result + iter * 16);

        swap = tile, tile = tile_buffer, tile_buffer = swap;
    }

    int offset2 = param.c * param.h * param.w / 4;
    int begin_idx = param.c * param.h * param.w * 4 * by;
    int row = (bx / 4) * (param.h * param.w / 128) * 16 + (bx * 4) % 16 + (tid / 32) * 16;
    short col = tid % 32;

    _Float16 *load_result = param.transform_pin + begin_idx + row * 32 + col;
    #pragma unroll
    for (short i = 0; i < 16; i++) {
        #pragma unroll
        for(short j = 0; j < 4; j++){
            load_result[j * 32] = result[j * 16 + i];
        }
        load_result += offset2;
    }
}


__global__ void BTdB3(mykernelParamType param){
    short bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
    short tid = threadIdx.y * blockDim.x + threadIdx.x;
    short warp_id = tid / 64, lane_id = tid % 64;

    _Float16 *input = param.pin + (bz * param.c + by) * param.h * param.w + bx * 63 * param.w;

    __shared__  _Float16 share_mem[128 * 65];

    reinterpret_cast<ulonglong4*>(share_mem)[tid * 2] = reinterpret_cast<ulonglong4*>(input)[tid * 2];
    reinterpret_cast<ulonglong4*>(share_mem)[tid * 2 + 1] = reinterpret_cast<ulonglong4*>(input)[tid * 2 + 1];

    if(tid < 128)    share_mem[8192 + tid] = input[8192 + tid];

    __syncthreads();

    short mask = 0xffff;
    init_mask(mask, lane_id, warp_id + bx * 32, param.w / 2);

    _Float16 *share = share_mem + warp_id * 2 * param.w + bx * param.w + 2 * lane_id;

    _Float16 tile_mem[32], result[128];
    _Float16 *tile = tile_mem, *tile_buffer = tile_mem + 16, *swap;

    prefetch_input(mask, param.w, share, tile);

    #pragma unroll
    for(short iter = 0;iter < 8; iter++){
        if(iter < 7){
            share += 8 * param.w;
            init_mask(mask, lane_id, warp_id + bx * 32 + 4 * (iter + 1), param.w / 2);
            prefetch_input(mask, param.w, share, tile_buffer);
        }
        calculate_transform_pin(tile, result + iter * 16);
        swap = tile, tile =tile_buffer, tile_buffer = swap;
    }
    int offset2 = param.c * param.h * param.w / 4;
    _Float16 *load_result = param.transform_pin + param.c * param.h * param.w * 4 * bz + (bx * 256 + by) * 512 + tid;
    #pragma unroll
    for(short i = 0; i < 16; i++){
        #pragma unroll
        for(short j = 0; j < 4; j++){
            load_result[j * 512 * 64] = result[j * 32 + i];
            load_result[j * 512 * 64 + 256] = result[j * 32 + i + 16];
        }
        load_result += offset2;
    }
}

#endif