#ifndef MATRIXMUL_H
#define MATRIXMUL_H
#include "mykernelParamType.cuh"

typedef _Float16 half8_ __attribute__((ext_vector_type(8)));
typedef _Float16 half4_ __attribute__((ext_vector_type(4)));
typedef float float4_ __attribute__((ext_vector_type(4)));

union RegisterUnion
{
  half8_ vector8;
  struct
  {
    half4_ vector_front;
    half4_ vector_rear;
  };
};


__device__  __inline__  void init_result(float4_& fragC00, float4_& fragC01, float4_& fragC10, float4_& fragC11){
    fragC00 = {0.0f, 0.0f, 0.0f, 0.0f};
    fragC01 = {0.0f, 0.0f, 0.0f, 0.0f};
    fragC10 = {0.0f, 0.0f, 0.0f, 0.0f};
    fragC11 = {0.0f, 0.0f, 0.0f, 0.0f};
}


__device__  __inline__  void mul_add(RegisterUnion& fragA, RegisterUnion& fragB, 
        float4_& fragC00, float4_& fragC01, float4_& fragC10, float4_& fragC11){

    asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC00), "+v"(fragA.vector_front), "+v"(fragB.vector_front));
    asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC01), "+v"(fragA.vector_rear), "+v"(fragB.vector_front));
    asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC10), "+v"(fragA.vector_front), "+v"(fragB.vector_rear));
    asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC11), "+v"(fragA.vector_rear), "+v"(fragB.vector_rear));
}


__device__  __inline__ void store_result(_Float16 *load_result, float4_& fragC00, float4_& fragC10, float4_& fragD00, float4_& fragD10, 
            int idx, int weight, int output_row, int output_col){

    idx += weight * output_row + output_col;
    load_result[idx] = (_Float16)fragC00.x;
    load_result[idx + 4] = (_Float16)fragC00.y;
    load_result[idx + 8] = (_Float16)fragC00.z;
    load_result[idx + 12] = (_Float16)fragC00.w;

    load_result[idx + 16] = (_Float16)fragC10.x;
    load_result[idx + 20] = (_Float16)fragC10.y;
    load_result[idx + 24] = (_Float16)fragC10.z;
    load_result[idx + 28] = (_Float16)fragC10.w;

    load_result[idx + 32] = (_Float16)fragD00.x;
    load_result[idx + 36] = (_Float16)fragD00.y;
    load_result[idx + 40] = (_Float16)fragD00.z;
    load_result[idx + 44] = (_Float16)fragD00.w;

    load_result[idx + 48] = (_Float16)fragD10.x;
    load_result[idx + 52] = (_Float16)fragD10.y;
    load_result[idx + 56] = (_Float16)fragD10.z;
    load_result[idx + 60] = (_Float16)fragD10.w;
}


__device__  __inline__ void store_result1(_Float16 *load_result, float4_& fragC00, float4_& fragC10, float4_& fragD00, float4_& fragD10, 
            float4_& fragZ00, float4_& fragZ10, float4_& fragW00, float4_& fragW10, int weight, int output_row, int output_col){

    int idx = weight * output_row + output_col;
    if(output_row < 27){
        load_result[idx] = (_Float16)fragC00.x;
        load_result[idx + 4] = (_Float16)fragC00.y;
        load_result[idx + 8] = (_Float16)fragC00.z;
        load_result[idx + 12] = (_Float16)fragC00.w;

        load_result[idx + 16] = (_Float16)fragC10.x;
        load_result[idx + 20] = (_Float16)fragC10.y;
        load_result[idx + 24] = (_Float16)fragC10.z;
        load_result[idx + 28] = (_Float16)fragC10.w;

        load_result[idx + 32] = (_Float16)fragD00.x;
        load_result[idx + 36] = (_Float16)fragD00.y;
        load_result[idx + 40] = (_Float16)fragD00.z;
        load_result[idx + 44] = (_Float16)fragD00.w;

        load_result[idx + 48] = (_Float16)fragD10.x;
        load_result[idx + 52] = (_Float16)fragD10.y;
        load_result[idx + 56] = (_Float16)fragD10.z;
        load_result[idx + 60] = (_Float16)fragD10.w;

        load_result[idx + 64] = (_Float16)fragZ00.x;
        load_result[idx + 68] = (_Float16)fragZ00.y;
        load_result[idx + 72] = (_Float16)fragZ00.z;
        load_result[idx + 76] = (_Float16)fragZ00.w;

        load_result[idx + 80] = (_Float16)fragZ10.x;
        load_result[idx + 84] = (_Float16)fragZ10.y;
        load_result[idx + 88] = (_Float16)fragZ10.z;
        load_result[idx + 92] = (_Float16)fragZ10.w;

        load_result[idx + 96] = (_Float16)fragW00.x;
        load_result[idx + 100] = (_Float16)fragW00.y;
        load_result[idx + 104] = (_Float16)fragW00.z;
        load_result[idx + 108] = (_Float16)fragW00.w;

        load_result[idx + 112] = (_Float16)fragW10.x;
        load_result[idx + 116] = (_Float16)fragW10.y;
        load_result[idx + 120] = (_Float16)fragW10.z;
        load_result[idx + 124] = (_Float16)fragW10.w;
    }
}


__device__  __inline__ void store_result3(_Float16 *load_result, float4_& fragC00, float4_& fragC01, float4_& fragC10, float4_& fragC11, 
            int weight, int output_row, int output_col){

    int idx = weight * output_row + output_col;
    load_result[idx] = (_Float16)fragC00.x;
    load_result[idx + 4] = (_Float16)fragC00.y;
    load_result[idx + 8] = (_Float16)fragC00.z;
    load_result[idx + 12] = (_Float16)fragC00.w;

    load_result[idx + 16] = (_Float16)fragC10.x;
    load_result[idx + 20] = (_Float16)fragC10.y;
    load_result[idx + 24] = (_Float16)fragC10.z;
    load_result[idx + 28] = (_Float16)fragC10.w;

    idx += weight * 16;
    load_result[idx] = (_Float16)fragC01.x;
    load_result[idx + 4] = (_Float16)fragC01.y;
    load_result[idx + 8] = (_Float16)fragC01.z;
    load_result[idx + 12] = (_Float16)fragC01.w;

    load_result[idx + 16] = (_Float16)fragC11.x;
    load_result[idx + 20] = (_Float16)fragC11.y;
    load_result[idx + 24] = (_Float16)fragC11.z;
    load_result[idx + 28] = (_Float16)fragC11.w;
}


__global__ void matrixMul1(mykernelParamType param) {
    short bx = blockIdx.x, by = blockIdx.y;
    short tid = threadIdx.y * blockDim.x + threadIdx.x;
    short warp_id = tid / 64, lane_id = tid % 64;

    // 共享内存分配
    __shared__ _Float16 share_mem[32 * 16 + 512 * 16]; 
    _Float16 *matrix_A = share_mem, *matrix_B = share_mem + 32 * 16;

    matrix_A[tid * 2] = 0.0f;
    matrix_A[tid * 2 + 1] = 0.0f;

    _Float16 *load_filter = param.transform_filter;
    _Float16 *load_pin = param.transform_pin;
    
    RegisterUnion fragA;
    RegisterUnion fragB, fragC;

    float4_ fragX00, fragX01, fragX10, fragX11;
    float4_ fragY00, fragY01, fragY10, fragY11;
    float4_ fragZ00, fragZ01, fragZ10, fragZ11;
    float4_ fragW00, fragW01, fragW10, fragW11;

    init_result(fragX00, fragX01, fragX10, fragX11);
    init_result(fragY00, fragY01, fragY10, fragY11);
    init_result(fragZ00, fragZ01, fragZ10, fragZ11);
    init_result(fragW00, fragW01, fragW10, fragW11);

    short row = tid / 16, col = tid % 16;

    load_filter += (by % 16) * param.k * param.c;  
    load_pin += by * param.c * param.h * param.w / 4 + bx * 512 * 16;

    int lds_read_A_offset1 = lane_id * 8 * sizeof(_Float16);
    int lds_read_B_offset1 = lane_id * 8 * sizeof(_Float16) + 32 * 16 * sizeof(_Float16) + warp_id * 128 * 16 * sizeof(_Float16);
    int lds_read_B_offset2 = lds_read_B_offset1 + 32 * 16 * sizeof(_Float16);
    int lds_read_B_offset3 = lds_read_B_offset2 + 32 * 16 * sizeof(_Float16);
    int lds_read_B_offset4 = lds_read_B_offset3 + 32 * 16 * sizeof(_Float16);

    short iter = 0;
    do{
        __syncthreads();   
        #pragma unroll
        for (short i = 0; i < 2; i++) {
            if(col * 2 + i < 27){
                matrix_A[row * 32 + col * 2 + i] = load_filter[row * param.k + col * 2 + i];
            }
        }

        reinterpret_cast<ulonglong4*>(matrix_B)[tid * 2] = reinterpret_cast<ulonglong4*>(load_pin)[tid * 2];
        reinterpret_cast<ulonglong4*>(matrix_B)[tid * 2 + 1] = reinterpret_cast<ulonglong4*>(load_pin)[tid * 2 + 1];

        __syncthreads();

        load_filter += param.k * 16;
        load_pin += param.h * param.w * 4;

        asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragA.vector8), "+v"(lds_read_A_offset1));
        asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragB.vector8), "+v"(lds_read_B_offset1));
        
        asm volatile("s_waitcnt lgkmcnt(0)\n\t");
        asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragC.vector8), "+v"(lds_read_B_offset2));
        
        mul_add(fragA, fragB, fragX00, fragX01, fragX10, fragX11);

        asm volatile("s_waitcnt lgkmcnt(0)\n\t");
        asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragB.vector8), "+v"(lds_read_B_offset3));

        mul_add(fragA, fragC, fragY00, fragY01, fragY10, fragY11);

        asm volatile("s_waitcnt lgkmcnt(0)\n\t");
        asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragC.vector8), "+v"(lds_read_B_offset4));

        mul_add(fragA, fragB, fragZ00, fragZ01, fragZ10, fragZ11);

        asm volatile("s_waitcnt lgkmcnt(0)\n\t");

        mul_add(fragA, fragC, fragW00, fragW01, fragW10, fragW11);

        iter++;
    }while(iter < param.c / 16);

    // 结果存储
    _Float16 *result = param.mid_result + by * param.k * param.h * param.w / 4 + (bx * 2 + warp_id / 2) * 256 * 27 + (warp_id % 2) * 128;
    short output_row = lane_id & 15;
    short output_col = lane_id >> 4;
    store_result1(result, fragX00, fragX10, fragY00, fragY10, fragZ00, fragZ10, fragW00, fragW10, 256, output_row, output_col);
    output_row += 16;
    store_result1(result, fragX01, fragX11, fragY01, fragY11, fragZ01, fragZ11, fragW01, fragW11, 256, output_row, output_col);
}


__launch_bounds__(512)
__global__ void matrixMul3(mykernelParamType param) {
    short bx = blockIdx.x, by = blockIdx.y;
    short tid = threadIdx.y * blockDim.x + threadIdx.x;
    short warp_id = tid / 64, lane_id = tid % 64;

    // 共享内存分配
    __shared__ _Float16 share_mem[64 * 16 + 512 * 16]; 
    _Float16 *matrix_A = share_mem, *matrix_B = share_mem + 64 * 16;

    _Float16 *load_filter = param.transform_filter;
    _Float16 *load_pin = param.transform_pin;
    
    // 寄存器分配
    RegisterUnion fragA, fragB;
    RegisterUnion fragC, fragD;

    float4_ fragX00, fragX01, fragX10, fragX11;
    float4_ fragY00, fragY01, fragY10, fragY11;
    float4_ fragZ00, fragZ01, fragZ10, fragZ11;
    float4_ fragW00, fragW01, fragW10, fragW11;

    init_result(fragX00, fragX01, fragX10, fragX11);
    init_result(fragY00, fragY01, fragY10, fragY11);
    init_result(fragZ00, fragZ01, fragZ10, fragZ11);
    init_result(fragW00, fragW01, fragW10, fragW11);

    short row = tid / 32, col = tid % 32;
    int offset1 = (col / 2) * 2 * 15;
    // 把转换后的卷积核矩阵从全局内存拷贝到共享内存
    load_filter += (by % 16) * param.k * param.c;  
    load_pin += by * param.c * param.h * param.w / 4 + bx * 512 * 64;

    int lds_read_A_offset1 = lane_id * 8 * sizeof(_Float16);
    int lds_read_A_offset2 = lds_read_A_offset1 + 32 * 16 * sizeof(_Float16);
    int lds_read_B_offset1 = lane_id * 8 * sizeof(_Float16) + 64 * 16 * sizeof(_Float16) + warp_id * 64 * 16 * sizeof(_Float16);
    int lds_read_B_offset2 = lds_read_B_offset1 + 32 * 16 * sizeof(_Float16);

    short iter = 0;
    do{
        __syncthreads();
        reinterpret_cast<ushort2*>(matrix_A)[tid] = reinterpret_cast<ushort2*>(load_filter)[tid];
        reinterpret_cast<ulonglong4*>(matrix_B)[row * 2 + col + offset1] = reinterpret_cast<ulonglong4*>(load_pin)[row * 32 + col];
        __syncthreads();

        load_filter += param.k * 16;
        load_pin += 512 * 16;

        asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragA.vector8), "+v"(lds_read_A_offset1));
        asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragC.vector8), "+v"(lds_read_B_offset1));
        
        asm volatile("s_waitcnt lgkmcnt(0)\n\t");
        asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragD.vector8), "+v"(lds_read_B_offset2));
        
        mul_add(fragA, fragC, fragX00, fragX01, fragX10, fragX11);

        asm volatile("s_waitcnt lgkmcnt(0)\n\t");
        asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragB.vector8), "+v"(lds_read_A_offset2));
        mul_add(fragA, fragD, fragY00, fragY01, fragY10, fragY11);

        asm volatile("s_waitcnt lgkmcnt(0)\n\t");
        mul_add(fragB, fragD, fragW00, fragW01, fragW10, fragW11);

        asm volatile("s_waitcnt lgkmcnt(0)\n\t");
        mul_add(fragB, fragC, fragZ00, fragZ01, fragZ10, fragZ11);

        iter++;
    }while(iter < param.c / 16);

    // 结果存储
    _Float16 *result = param.mid_result + by * param.k * param.h * param.w / 4 + bx * 512 * 64 + (warp_id / 4) * 256 * 64 + (warp_id % 4) * 64;
    short output_row = lane_id & 15;
    short output_col = lane_id >> 4;
    store_result3(result     , fragX00, fragX01, fragX10, fragX11, 256, output_row, output_col);
    store_result3(result + 32, fragY00, fragY01, fragY10, fragY11, 256, output_row, output_col);

    result += 32 * 256;
    store_result3(result     , fragZ00, fragZ01, fragZ10, fragZ11, 256, output_row, output_col);
    store_result3(result + 32, fragW00, fragW01, fragW10, fragW11, 256, output_row, output_col);
}


__launch_bounds__(512)
__global__ void matrixMul4(mykernelParamType param) {
    short bx = blockIdx.x, by = blockIdx.y;
    short tid = threadIdx.y * blockDim.x + threadIdx.x;
    short warp_id = tid / 64, lane_id = tid % 64;

    // 共享内存分配
    __shared__ _Float16 share_mem[128 * 16 * 3]; 
    _Float16 *matrix_A = share_mem, *matrix_B = share_mem + 128 * 16;

    _Float16 *load_filter = param.transform_filter;
    _Float16 *load_pin = param.transform_pin;
    
    // 寄存器分配
    RegisterUnion fragA, fragB;
    RegisterUnion fragC, fragD;

    float4_ fragX00, fragX01, fragX10, fragX11;
    float4_ fragY00, fragY01, fragY10, fragY11;
    float4_ fragZ00, fragZ01, fragZ10, fragZ11;
    float4_ fragW00, fragW01, fragW10, fragW11;

    init_result(fragX00, fragX01, fragX10, fragX11);
    init_result(fragY00, fragY01, fragY10, fragY11);
    init_result(fragZ00, fragZ01, fragZ10, fragZ11);
    init_result(fragW00, fragW01, fragW10, fragW11);

    short row = tid / 4, col = tid % 4;
    int offset1 = (row / 32) * 32 * 15;
    // 把转换后的卷积核矩阵从全局内存拷贝到共享内存
    load_filter += (by % 16) * param.k * param.c + bx * 128 * param.c;  
    load_pin += by * param.c * param.h * param.w / 4;

    int lds_read_A_offset1 = lane_id * 8 * sizeof(_Float16) + (warp_id / 4) * 64 * 16 * sizeof(_Float16);
    int lds_read_A_offset2 = lds_read_A_offset1 + 32 * 16 * sizeof(_Float16);
    int lds_read_B_offset1 = lane_id * 8 * sizeof(_Float16) + 128 * 16 * sizeof(_Float16) + (warp_id % 4) * 64 * 16 * sizeof(_Float16);
    int lds_read_B_offset2 = lds_read_B_offset1 + 32 * 16 * sizeof(_Float16);

    short iter = 0;
    do{
        __syncthreads();   
        #pragma unroll
        for (short i = 0; i < 4; i++) {
            matrix_A[(col * 4 + i) * 32 + row + offset1] = load_filter[row * param.c + col * 4 + i];
        }

        reinterpret_cast<ulonglong2*>(matrix_B)[tid] = reinterpret_cast<ulonglong2*>(load_pin)[tid];

        __syncthreads();

        load_filter += 16;
        load_pin += param.h * param.w * 4;

        asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragA.vector8), "+v"(lds_read_A_offset1));
        asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragC.vector8), "+v"(lds_read_B_offset1));
        
        asm volatile("s_waitcnt lgkmcnt(0)\n\t");
        
        asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragD.vector8), "+v"(lds_read_B_offset2));
        
        mul_add(fragA, fragC, fragX00, fragX01, fragX10, fragX11);

        asm volatile("s_waitcnt lgkmcnt(0)\n\t");
        asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragB.vector8), "+v"(lds_read_A_offset2));
        mul_add(fragA, fragD, fragY00, fragY01, fragY10, fragY11);

        asm volatile("s_waitcnt lgkmcnt(0)\n\t");
        mul_add(fragB, fragD, fragW00, fragW01, fragW10, fragW11);

        asm volatile("s_waitcnt lgkmcnt(0)\n\t");
        mul_add(fragB, fragC, fragZ00, fragZ01, fragZ10, fragZ11);

        iter++;
    }while(iter < param.c / 16);

    // 结果存储
    _Float16 *load_result = param.mid_result + by * param.k * param.h * param.w / 4;
    int idx = (bx * 2 + (warp_id / 4)) * param.h * param.w * 16 + (warp_id % 4) * 64;

    short output_row = lane_id & 15;
    short output_col = lane_id >> 4;
    store_result(load_result, fragX00, fragX10, fragY00, fragY10, idx, param.h * param.w / 4, output_row, output_col);
    load_result += param.h * param.w * 4;
    store_result(load_result, fragX01, fragX11, fragY01, fragY11, idx, param.h * param.w / 4, output_row, output_col);
    load_result += param.h * param.w * 4;
    store_result(load_result, fragZ00, fragZ10, fragW00, fragW10, idx, param.h * param.w / 4, output_row, output_col);
    load_result += param.h * param.w * 4;
    store_result(load_result, fragZ01, fragZ11, fragW01, fragW11, idx, param.h * param.w / 4, output_row, output_col);
}


__global__ void matrixMul5(mykernelParamType param) {
    short bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
    short tid = threadIdx.y * blockDim.x + threadIdx.x;
    short warp_id = tid / 64, lane_id = tid % 64;

    // 共享内存分配
    __shared__ _Float16 share_mem[128 * 16 * 2]; 
    _Float16 *matrix_A = share_mem, *matrix_B = share_mem + 128 * 16;

    _Float16 *load_filter = param.transform_filter + (bz % 16) * param.k * param.c + bx * 128 * 16;  
    _Float16 *load_pin = param.transform_pin + bz * param.c * param.h * param.w / 4 + by * 128 * 16;
    
    // 寄存器分配
    RegisterUnion fragA, fragB;
    RegisterUnion fragC, fragD;

    float4_ fragX00, fragX01, fragX10, fragX11;
    float4_ fragY00, fragY01, fragY10, fragY11;
    float4_ fragZ00, fragZ01, fragZ10, fragZ11;
    float4_ fragW00, fragW01, fragW10, fragW11;

    init_result(fragX00, fragX01, fragX10, fragX11);
    init_result(fragY00, fragY01, fragY10, fragY11);
    init_result(fragZ00, fragZ01, fragZ10, fragZ11);
    init_result(fragW00, fragW01, fragW10, fragW11);

    int lds_read_A_offset1 = lane_id * 8 * sizeof(_Float16) + (warp_id / 2) * 64 * 16 * sizeof(_Float16);
    int lds_read_A_offset2 = lds_read_A_offset1 + 32 * 16 * sizeof(_Float16);
    int lds_read_B_offset1 = lane_id * 8 * sizeof(_Float16) + 128 * 16 * sizeof(_Float16) + (warp_id % 2) * 64 * 16 * sizeof(_Float16);
    int lds_read_B_offset2 = lds_read_B_offset1 + 32 * 16 * sizeof(_Float16);

    short iter = 0;
    do{
        __syncthreads();   
        reinterpret_cast<ulonglong2*>(matrix_A)[tid] = reinterpret_cast<ulonglong2*>(load_filter)[tid];
        reinterpret_cast<ulonglong2*>(matrix_B)[tid] = reinterpret_cast<ulonglong2*>(load_pin)[tid];
        __syncthreads();

        load_filter += param.k * 16;
        load_pin += param.h * param.w * 4;

        asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragA.vector8), "+v"(lds_read_A_offset1));
        asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragC.vector8), "+v"(lds_read_B_offset1));
        
        asm volatile("s_waitcnt lgkmcnt(0)\n\t");
        asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragD.vector8), "+v"(lds_read_B_offset2));
        
        mul_add(fragA, fragC, fragX00, fragX01, fragX10, fragX11);

        asm volatile("s_waitcnt lgkmcnt(0)\n\t");
        asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragB.vector8), "+v"(lds_read_A_offset2));
        mul_add(fragA, fragD, fragY00, fragY01, fragY10, fragY11);

        asm volatile("s_waitcnt lgkmcnt(0)\n\t");
        mul_add(fragB, fragD, fragW00, fragW01, fragW10, fragW11);

        asm volatile("s_waitcnt lgkmcnt(0)\n\t");
        mul_add(fragB, fragC, fragZ00, fragZ01, fragZ10, fragZ11);

        iter++;
    }while(iter < param.c / 16);

    // 结果存储
    _Float16 *load_result = param.mid_result + bz * param.k * param.h * param.w / 4;
    int idx = (bx * gridDim.y + by) * 128 * 128 + (warp_id / 2) * 128 * 64 + (warp_id % 2) * 64;

    short output_row = lane_id & 15;
    short output_col = lane_id >> 4;
    store_result(load_result, fragX00, fragX10, fragY00, fragY10, idx, 128, output_row, output_col);
    load_result += 128 * 16;
    store_result(load_result, fragX01, fragX11, fragY01, fragY11, idx, 128, output_row, output_col);
    load_result += 128 * 16;
    store_result(load_result, fragZ00, fragZ10, fragW00, fragW10, idx, 128, output_row, output_col);
    load_result += 128 * 16;
    store_result(load_result, fragZ01, fragZ11, fragW01, fragW11, idx, 128, output_row, output_col);
}


__global__ void matrixMul6(mykernelParamType param) {
    short bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
    short tid = threadIdx.y * blockDim.x + threadIdx.x;
    short warp_id = tid / 64, lane_id = tid % 64;

    // 共享内存分配
    __shared__ _Float16 share_mem[32 * 16 + 64 * 16]; 
    _Float16 *matrix_A = share_mem, *matrix_B = share_mem + 32 * 16;

    #pragma unroll
    for(int i = 0; i < 8; i++){
        matrix_A[tid * 8 + i] = 0.0f;
    }

    _Float16 *load_filter = param.transform_filter;
    _Float16 *load_pin = param.transform_pin;
    
    RegisterUnion fragA;
    RegisterUnion fragB;

    float4_ fragX00, fragX10;
    fragX00 = {0.0f, 0.0f, 0.0f, 0.0f};
    fragX10 = {0.0f, 0.0f, 0.0f, 0.0f};

    short row = tid / 2, col = tid % 2;

    load_filter += (bz % 16) * param.k * param.c;  
    load_pin += bz * param.c * param.h * param.w / 4 + bx * 64 * 16;

    int lds_read_A_offset = lane_id * 8 * sizeof(_Float16);
    int lds_read_B_offset = lds_read_A_offset + (warp_id + 1) * 32 * 16 * sizeof(_Float16);

    short iter = 0;
    do{
        __syncthreads();
        if(tid < 32){
            matrix_A[row * 32 + col * 2] = load_filter[row * param.k + col * 2];
            matrix_A[row * 32 + col * 2 + 1] = load_filter[row * param.k + col * 2 + 1];
        }

        reinterpret_cast<ulonglong2*>(matrix_B)[tid] = reinterpret_cast<ulonglong2*>(load_pin)[tid];
        __syncthreads();

        load_filter += param.k * 16;
        load_pin += param.h * param.w * 4;

        asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragA.vector8), "+v"(lds_read_A_offset));
        asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragB.vector8), "+v"(lds_read_B_offset));
        
        asm volatile("s_waitcnt lgkmcnt(0)\n\t");   

        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragX00), "+v"(fragA.vector_front), "+v"(fragB.vector_front));
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragX10), "+v"(fragA.vector_front), "+v"(fragB.vector_rear));

        iter++;
    }while(iter < param.c / 16);

    // 结果存储
    short output_row = lane_id & 15;
    short output_col = lane_id >> 4;
    if(output_row < 4){
        int weight = param.h * param.w / 4;
        _Float16 *result = param.mid_result + (bz * param.k + output_row) * weight + (bx * 2 + warp_id) * 32 + output_col;

        result[0] = (_Float16)fragX00.x;
        result[4] = (_Float16)fragX00.y;
        result[8] = (_Float16)fragX00.z;
        result[12] = (_Float16)fragX00.w;

        result[16] = (_Float16)fragX10.x;
        result[20] = (_Float16)fragX10.y;
        result[24] = (_Float16)fragX10.z;
        result[28] = (_Float16)fragX10.w;
    }
}


#endif
