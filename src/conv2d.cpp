#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include "preprocessing.cuh"
#include "matrixmul.cuh"
#include "conv2d.h"


/*选手需要返回自定义kernel入参结构体的size*/
int getParamsize(__in__ problem_t* problem, __out__ int* paramSize)
{
    *paramSize = sizeof(mykernelParamType);

    return 0;
}


__global__ void store(mykernelParamType param){
    //         AT matrix              output tile          A matrix
    // *  |   1,   1,   1,  0|   |  x1,  x2,  x3,  x4|     | 1,   0|
    // *  |   0,   1,  -1, -1|   |  x5,  x6,  x7,  x8|     | 1,   1|
    // *                         |  x9, x10, x11, x12|     | 1,  -1|
    // *                         | x13, x14, x15, x16|     | 0,  -1|
    short bx  = blockIdx.x, by = blockIdx.y;
    short bdx = blockDim.x;
    short tx  = threadIdx.x;

    _Float16 *load_input = param.mid_result;
    
    load_input += by * param.k * param.h * param.w * 4 + bx * bdx;
    int offset = param.k * param.h * param.w / 4;
    _Float16 tile[16], result[4];
    #pragma unroll
    for(int i = 0; i < 16; i++){
        tile[i] = load_input[tx];
        load_input += offset;
    }
    result[0] = tile[0] + tile[1] + tile[2] + tile[4] + tile[5] + tile[6] + tile[8] + tile[9] + tile[10] ; 
    result[1] = tile[1] - tile[2] - tile[3] + tile[5] - tile[6] - tile[7] + tile[9] - tile[10] - tile[11];
    result[2] = tile[4] + tile[5] + tile[6] - tile[8] - tile[9] - tile[10] - tile[12] - tile[13] - tile[14]; 
    result[3] = tile[5] - tile[6] - tile[7] - tile[9] + tile[10] + tile[11] - tile[13] + tile[14] + tile[15];

    int row = (bx * bdx + tx) / (param.h / 2), col = (bx * bdx + tx) % (param.h / 2);
    int idx = by * param.k * param.Oh * param.Ow + row * param.Ow * 2 + col * 2;
    _Float16 *load_result = param.pout + idx;

    reinterpret_cast<ushort2*>(load_result)[0] = reinterpret_cast<ushort2*>(result)[0];
    reinterpret_cast<ushort2*>(load_result)[param.Ow / 2] = reinterpret_cast<ushort2*>(result)[1];
}


__global__ void store1(mykernelParamType param){
    short bx  = blockIdx.x, by = blockIdx.y;
    short tid = threadIdx.y * blockDim.x + threadIdx.x;

    _Float16 *load_input = param.mid_result;
    
    load_input += by * param.k * param.h * param.w * 4 + (bx / 2) * 256 + (bx % 2) * 256 * param.k * 2;
    int offset = param.k * param.h * param.w / 4;
    _Float16 tile[2][16], result_mem[8];
    _Float16 *result = result_mem, *result_buffer = result_mem + 4;
    _Float16 *swap;

    #pragma unroll
    for(short i = 0; i < 16; i++){
        tile[0][i] = load_input[tid];
        tile[1][i] = load_input[tid + 256 * 27];
        load_input += offset;
    }
    int row, col, idx;
    row = ((bx % 2) * 512 + tid) / (param.Oh / 2), col = ((bx % 2) * 512 + tid) % (param.Oh / 2);
    idx = (by * param.k + bx / 2) * param.Oh * param.Ow + row * param.Ow * 2 + col * 2;
    _Float16 *load_result = param.pout + idx;
    #pragma unroll
    for(short i = 0; i < 2; i++){
        result[0] = tile[i][0] + tile[i][1] + tile[i][2] + tile[i][4] + tile[i][5] + tile[i][6] + tile[i][8] + tile[i][9] + tile[i][10] ; 
        result[1] = tile[i][1] - tile[i][2] - tile[i][3] + tile[i][5] - tile[i][6] - tile[i][7] + tile[i][9] - tile[i][10] - tile[i][11];
        result[2] = tile[i][4] + tile[i][5] + tile[i][6] - tile[i][8] - tile[i][9] - tile[i][10] - tile[i][12] - tile[i][13] - tile[i][14]; 
        result[3] = tile[i][5] - tile[i][6] - tile[i][7] - tile[i][9] + tile[i][10] + tile[i][11] - tile[i][13] + tile[i][14] + tile[i][15];

        swap = result_buffer, result_buffer = result, result = swap;

        reinterpret_cast<ushort2*>(load_result)[0] = reinterpret_cast<ushort2*>(result_buffer)[0];
        reinterpret_cast<ushort2*>(load_result)[param.Ow / 2] = reinterpret_cast<ushort2*>(result_buffer)[1];

        load_result += 16 * param.Ow;
    }
}


__global__ void store2(mykernelParamType param){
    short bx  = blockIdx.x, by = blockIdx.y;
    short bd  = blockDim.x * blockDim.y;
    short tid = threadIdx.y * blockDim.x + threadIdx.x;

    _Float16 *load_input = param.mid_result;
    load_input += by * param.k * param.h * param.w * 4 + (bx / 32) * 256 * 128 + (bx % 32) * 128 * 4 + (tid / 128) * 128 * 128 + tid % 128; 

    int offset = param.k * param.h * param.w / 4;
    _Float16 tile[4][16], result_mem[8];
    _Float16 *result = result_mem, *result_buffer = result_mem + 4;
    _Float16 *swap;

    #pragma unroll
    for(short i = 0; i < 16; i++){
        tile[0][i] = load_input[0];
        tile[1][i] = load_input[128];
        tile[2][i] = load_input[256];
        tile[3][i] = load_input[384];
        load_input += offset;
    }
    int row, col, idx;
    row = ((bx * bd) * 4 + tid) / (param.h / 2), col = ((bx * bd) * 4 + tid) % (param.h / 2);
    idx = by * param.k * param.Oh * param.Ow + row * param.Ow * 2 + col * 2;
    _Float16 *load_result = param.pout + idx;

    #pragma unroll
    for(short i = 0; i < 4; i++){
        result[0] = tile[i][0] + tile[i][1] + tile[i][2] + tile[i][4] + tile[i][5] + tile[i][6] + tile[i][8] + tile[i][9] + tile[i][10] ; 
        result[1] = tile[i][1] - tile[i][2] - tile[i][3] + tile[i][5] - tile[i][6] - tile[i][7] + tile[i][9] - tile[i][10] - tile[i][11];
        result[2] = tile[i][4] + tile[i][5] + tile[i][6] - tile[i][8] - tile[i][9] - tile[i][10] - tile[i][12] - tile[i][13] - tile[i][14]; 
        result[3] = tile[i][5] - tile[i][6] - tile[i][7] - tile[i][9] + tile[i][10] + tile[i][11] - tile[i][13] + tile[i][14] + tile[i][15];

        swap = result_buffer, result_buffer = result, result = swap;

        reinterpret_cast<ushort2*>(load_result)[0] = reinterpret_cast<ushort2*>(result_buffer)[0];
        reinterpret_cast<ushort2*>(load_result)[param.Ow / 2] = reinterpret_cast<ushort2*>(result_buffer)[1];

        load_result += param.Oh * param.Ow;
    }
}


__global__ void store3(mykernelParamType param){
    short bx  = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
    short tid = threadIdx.y * blockDim.x + threadIdx.x;

    _Float16 *load_input   = param.mid_result + bz * param.k * param.h * param.w * 4 + (bx * 64 * 8 + by) * 256;
    _Float16 *load_output = param.pout + (bz * param.k + by) * param.Oh * param.Ow + (bx * 4 * 8 + tid / 64) * param.Ow * 2 + (tid % 64) * 2;
    
    int offset = param.k * param.h * param.w / 8;
    _Float16 tile[8][16], result[4];
    #pragma unroll
    for(short i = 0; i < 16; i++){
        #pragma unroll
        for(short j = 0; j < 8; j++){
            tile[j][i] = load_input[tid];
            load_input += 256 * 64;
        }
        load_input += offset;
    }

    #pragma unroll
    for(short iter = 0; iter < 8; iter++){
        result[0] = tile[iter][0] + tile[iter][1] + tile[iter][2] + tile[iter][4] + tile[iter][5] + tile[iter][6] + tile[iter][8] + tile[iter][9] + tile[iter][10] ; 
        result[1] = tile[iter][1] - tile[iter][2] - tile[iter][3] + tile[iter][5] - tile[iter][6] - tile[iter][7] + tile[iter][9] - tile[iter][10] - tile[iter][11];
        result[2] = tile[iter][4] + tile[iter][5] + tile[iter][6] - tile[iter][8] - tile[iter][9] - tile[iter][10] - tile[iter][12] - tile[iter][13] - tile[iter][14]; 
        result[3] = tile[iter][5] - tile[iter][6] - tile[iter][7] - tile[iter][9] + tile[iter][10] + tile[iter][11] - tile[iter][13] + tile[iter][14] + tile[iter][15];

        reinterpret_cast<ushort2*>(load_output)[0] = reinterpret_cast<ushort2*>(result)[0];
        reinterpret_cast<ushort2*>(load_output)[param.Ow / 2] = reinterpret_cast<ushort2*>(result)[1];

        load_output += param.Ow * 8;
    }
}


__global__ void store4(mykernelParamType param){
    short bx  = blockIdx.x, by = blockIdx.y;
    short bdx = blockDim.x * blockDim.y;
    short tid  = threadIdx.y * blockDim.x + threadIdx.x;

    _Float16 *load_input = param.mid_result;
    
    load_input += by * param.k * param.h * param.w * 4 + (bx * bdx) * 2 + tid;
    int offset = param.k * param.h * param.w / 4;
    _Float16 tile[2][16], result_mem[8];
    _Float16 *result = result_mem, *result_buffer = result_mem + 4;
    _Float16 *swap;

    #pragma unroll
    for(int i = 0; i < 16; i++){
        tile[0][i] = load_input[0];
        tile[1][i] = load_input[param.h * param.w / 4];
        load_input += offset;
    }
    int row, col, idx;
    row = (bx * bdx * 2 + tid) / (param.h / 2), col = (bx * bdx * 2 + tid) % (param.h / 2);
    idx = by * param.k * param.Oh * param.Ow + row * param.Ow * 2 + col * 2;
    _Float16 *load_result = param.pout + idx;

    #pragma unroll
    for(int i = 0; i < 2; i++){
        result[0] = tile[i][0] + tile[i][1] + tile[i][2] + tile[i][4] + tile[i][5] + tile[i][6] + tile[i][8] + tile[i][9] + tile[i][10] ; 
        result[1] = tile[i][1] - tile[i][2] - tile[i][3] + tile[i][5] - tile[i][6] - tile[i][7] + tile[i][9] - tile[i][10] - tile[i][11];
        result[2] = tile[i][4] + tile[i][5] + tile[i][6] - tile[i][8] - tile[i][9] - tile[i][10] - tile[i][12] - tile[i][13] - tile[i][14]; 
        result[3] = tile[i][5] - tile[i][6] - tile[i][7] - tile[i][9] + tile[i][10] + tile[i][11] - tile[i][13] + tile[i][14] + tile[i][15];
    
        swap = result_buffer;
        result_buffer = result;
        result = swap;

        reinterpret_cast<ushort2*>(load_result)[0] = reinterpret_cast<ushort2*>(result_buffer)[0];
        reinterpret_cast<ushort2*>(load_result)[param.Ow / 2] = reinterpret_cast<ushort2*>(result_buffer)[1];

        load_result += param.Oh * param.Ow;
    }
}


__global__ void store5(mykernelParamType param){
    short bx  = blockIdx.x, by = blockIdx.y;
    short bd  = blockDim.x * blockDim.y;
    short tid = threadIdx.y * blockDim.x + threadIdx.x;

    _Float16 *load_input = param.mid_result;
    load_input += by * param.k * param.h * param.w * 4 + (bx / 128) * 128 * 128 * 8 + (bx % 128) * 128 + (tid / 128) * 128 * 128 + tid % 128;

    int offset = param.k * param.h * param.w / 4;
    _Float16 tile[4][16], result_mem[8];
    _Float16 *result = result_mem, *result_buffer = result_mem + 4;
    _Float16 *swap;

    #pragma unroll
    for(short i = 0; i < 16; i++){
        #pragma unroll
        for(short j = 0; j < 4; j++){
            tile[j][i] = load_input[j * 128 * 256];
        }
        load_input += offset;
    }
    int row, col, idx;
    row = ((bx * bd) * 4 + tid) / (param.h / 2), col = ((bx * bd) * 4 + tid) % (param.h / 2);
    idx = by * param.k * param.Oh * param.Ow + row * param.Ow * 2 + col * 2;
    _Float16 *load_result = param.pout + idx;

    #pragma unroll
    for(short i = 0; i < 4; i++){
        result[0] = tile[i][0] + tile[i][1] + tile[i][2] + tile[i][4] + tile[i][5] + tile[i][6] + tile[i][8] + tile[i][9] + tile[i][10] ; 
        result[1] = tile[i][1] - tile[i][2] - tile[i][3] + tile[i][5] - tile[i][6] - tile[i][7] + tile[i][9] - tile[i][10] - tile[i][11];
        result[2] = tile[i][4] + tile[i][5] + tile[i][6] - tile[i][8] - tile[i][9] - tile[i][10] - tile[i][12] - tile[i][13] - tile[i][14]; 
        result[3] = tile[i][5] - tile[i][6] - tile[i][7] - tile[i][9] + tile[i][10] + tile[i][11] - tile[i][13] + tile[i][14] + tile[i][15];

        swap = result_buffer, result_buffer = result, result = swap;

        reinterpret_cast<ushort2*>(load_result)[0] = reinterpret_cast<ushort2*>(result_buffer)[0];
        reinterpret_cast<ushort2*>(load_result)[param.Ow / 2] = reinterpret_cast<ushort2*>(result_buffer)[1];

        load_result += 16 * param.Ow;
    }
}


/*选手需要返回自己优化的kernel的grid信息与kernel函数的指针*/
int getkernelInfo(__in__ problem_t* problem, __out__  kernelInfo_t* kernelInfo, __in_out__ void* param)
{
    mykernelParamType* pArgs = (mykernelParamType*)param;

    unsigned int n = problem->n;
    unsigned int c = problem->c;
    unsigned int h = problem->h;
    unsigned int w = problem->w;
    unsigned int k = problem->k;
    unsigned int r = problem->r;
    unsigned int s = problem->s;
    unsigned int u = problem->u;
    unsigned int v = problem->v;
    unsigned int p = problem->p;
    unsigned int q = problem->q;

    unsigned int outh = (h - r + 2*p)/u + 1;
    unsigned int outw = (w - s + 2*q)/v + 1;

    if(n == 16 && c == 128){
        kernelInfo->blockx1   = k;                    //blockx  number
        kernelInfo->blocky1   = c / 128;                    //blocky  number
        kernelInfo->blockz1   = 1;                    //blockz  number
        kernelInfo->threadx1  = 8;                   //threadx number per block
        kernelInfo->thready1  = 16;                   //thready number per block
        kernelInfo->threadz1  = 1;                   //threadz number per block
        kernelInfo->dynmicLdsSize1 = 0;
        kernelInfo->kernelPtr1= (void*)GgGT;                 //kernel ptr

        kernelInfo->blockx2   = c;                    
        kernelInfo->blocky2   = n;                    
        kernelInfo->blockz2   = 1;                    
        kernelInfo->threadx2  = 32;                   
        kernelInfo->thready2  = 8;                   
        kernelInfo->threadz2  = 1;                   
        kernelInfo->dynmicLdsSize2 = 0;
        kernelInfo->kernelPtr2= (void*)BTdB1;

        kernelInfo->blockx3   = 2;    
        kernelInfo->blocky3   = 16 * n;
        kernelInfo->blockz3   = 1;                     
        kernelInfo->threadx3  = 16;                   
        kernelInfo->thready3  = 16;                   
        kernelInfo->threadz3  = 1;                   
        kernelInfo->dynmicLdsSize3 = 0;
        kernelInfo->kernelPtr3= (void*)matrixMul1;                 //kernel ptr

        kernelInfo->blockx4   = k * 2;                    
        kernelInfo->blocky4   = n;                    
        kernelInfo->blockz4   = 1;                    
        kernelInfo->threadx4  = 16;                   
        kernelInfo->thready4  = 16;                   
        kernelInfo->threadz4  = 1;                   
        kernelInfo->dynmicLdsSize4 = 0;
        kernelInfo->kernelPtr4= (void*)store1;                 //kernel ptr
    }
    else if(n == 16 && c == 256){
        kernelInfo->blockx2   = c / 4;                    
        kernelInfo->blocky2   = n;                    
        kernelInfo->blockz2   = 1;                    
        kernelInfo->threadx2  = 16;                   
        kernelInfo->thready2  = 16;                   
        kernelInfo->threadz2  = 1;                   
        kernelInfo->dynmicLdsSize2 = 0;
        kernelInfo->kernelPtr2= (void*)BTdB2;

        kernelInfo->blockx1   = k / 16;                    //blockx  number
        kernelInfo->blocky1   = c / 16;                    //blocky  number
        kernelInfo->blockz1   = 1;                    //blockz  number
        kernelInfo->threadx1  = 16;                   //threadx number per block
        kernelInfo->thready1  = 16;                   //thready number per block
        kernelInfo->threadz1  = 1;                   //threadz number per block
        kernelInfo->dynmicLdsSize1 = 0;
        kernelInfo->kernelPtr1= (void*)GgGT2;                 //kernel ptr

        kernelInfo->blockx3   = 2;    
        kernelInfo->blocky3   = 2;
        kernelInfo->blockz3   = 16 * n;                     
        kernelInfo->threadx3  = 16;                   
        kernelInfo->thready3  = 16;                   
        kernelInfo->threadz3  = 1;                   
        kernelInfo->dynmicLdsSize3 = 0;
        kernelInfo->kernelPtr3= (void*)matrixMul5;                 //kernel ptr

        kernelInfo->blockx4   = 64;                    
        kernelInfo->blocky4   = n;                    
        kernelInfo->blockz4   = 1;                    
        kernelInfo->threadx4  = 16;                   
        kernelInfo->thready4  = 16;                   
        kernelInfo->threadz4  = 1;                   
        kernelInfo->dynmicLdsSize4 = 0;
        kernelInfo->kernelPtr4= (void*)store2;                 //kernel ptr
    }
    else if(n == 16 && c == 64){
        kernelInfo->blockx1   = k / 16;                    
        kernelInfo->blocky1   = c / 16;                    
        kernelInfo->blockz1   = 1;                    
        kernelInfo->threadx1  = 16;                   
        kernelInfo->thready1  = 16;                   
        kernelInfo->threadz1  = 1;                   
        kernelInfo->dynmicLdsSize1 = 0;
        kernelInfo->kernelPtr1= (void*)GgGT2;                 

        kernelInfo->blockx2   = 2;                    
        kernelInfo->blocky2   = c;                    
        kernelInfo->blockz2   = n;                    
        kernelInfo->threadx2  = 64;                   
        kernelInfo->thready2  = 4;                   
        kernelInfo->threadz2  = 1;                   
        kernelInfo->dynmicLdsSize2 = 0;
        kernelInfo->kernelPtr2= (void*)BTdB3;

        kernelInfo->blockx3   = 8;    
        kernelInfo->blocky3   = 16 * n;
        kernelInfo->blockz3   = 1;                     
        kernelInfo->threadx3  = 32;                   
        kernelInfo->thready3  = 16;                   
        kernelInfo->threadz3  = 1;                   
        kernelInfo->dynmicLdsSize3 = 0;
        kernelInfo->kernelPtr3= (void*)matrixMul3;

        kernelInfo->blockx4   = 2;                    
        kernelInfo->blocky4   = k;                    
        kernelInfo->blockz4   = n;                    
        kernelInfo->threadx4  = 16;                   
        kernelInfo->thready4  = 16;                   
        kernelInfo->threadz4  = 1;                   
        kernelInfo->dynmicLdsSize4 = 0;
        kernelInfo->kernelPtr4= (void*)store3;                 //kernel ptr
    }
    else if(n == 2 && c == 1920){
        kernelInfo->blockx1   = k / 64;               //blockx  number
        kernelInfo->blocky1   = c / 48;               //blocky  number
        kernelInfo->blockz1   = 1;                    //blockz  number
        kernelInfo->threadx1  = 16;                    //threadx number per block
        kernelInfo->thready1  = 16;                   //thready number per block
        kernelInfo->threadz1  = 1;                    //threadz number per block
        kernelInfo->dynmicLdsSize1 = 0;
        kernelInfo->kernelPtr1= (void*)GgGT4;         //kernel ptr

        kernelInfo->blockx2   = c / 4;                    
        kernelInfo->blocky2   = n;                    
        kernelInfo->blockz2   = 1;                    
        kernelInfo->threadx2  = 16;                   
        kernelInfo->thready2  = 16;                   
        kernelInfo->threadz2  = 1;                   
        kernelInfo->dynmicLdsSize2 = 0;
        kernelInfo->kernelPtr2= (void*)BTdB2;

        kernelInfo->blockx3   = 5;    
        kernelInfo->blocky3   = 16 * n;
        kernelInfo->blockz3   = 1;                     
        kernelInfo->threadx3  = 32;                   
        kernelInfo->thready3  = 16;                   
        kernelInfo->threadz3  = 1;                   
        kernelInfo->dynmicLdsSize3 = 0;
        kernelInfo->kernelPtr3= (void*)matrixMul4;

        kernelInfo->blockx4   = 320;                    
        kernelInfo->blocky4   = n;                    
        kernelInfo->blockz4   = 1;                    
        kernelInfo->threadx4  = 16;                   
        kernelInfo->thready4  = 16;                   
        kernelInfo->threadz4  = 1;                   
        kernelInfo->dynmicLdsSize4 = 0;
        kernelInfo->kernelPtr4= (void*)store4;  
    }
    else if(n == 2 && c == 640){
        kernelInfo->blockx1   = k / 64;                    //blockx  number
        kernelInfo->blocky1   = c / 32;                    //blocky  number
        kernelInfo->blockz1   = 1;                    //blockz  number
        kernelInfo->threadx1  = 8;                   //threadx number per block
        kernelInfo->thready1  = 32;                   //thready number per block
        kernelInfo->threadz1  = 1;                   //threadz number per block
        kernelInfo->dynmicLdsSize1 = 0;
        kernelInfo->kernelPtr1= (void*)GgGT5;                 //kernel ptr

        kernelInfo->blockx2   = c;                    
        kernelInfo->blocky2   = n;                    
        kernelInfo->blockz2   = 1;                    
        kernelInfo->threadx2  = 32;                   
        kernelInfo->thready2  = 8;                   
        kernelInfo->threadz2  = 1;                   
        kernelInfo->dynmicLdsSize2 = 0;
        kernelInfo->kernelPtr2= (void*)BTdB1;

        kernelInfo->blockx3   = 5;    
        kernelInfo->blocky3   = 8;
        kernelInfo->blockz3   = 16 * n;                     
        kernelInfo->threadx3  = 16;                   
        kernelInfo->thready3  = 16;                   
        kernelInfo->threadz3  = 1;                   
        kernelInfo->dynmicLdsSize3 = 0;
        kernelInfo->kernelPtr3= (void*)matrixMul5;

        kernelInfo->blockx4   = 640;                    
        kernelInfo->blocky4   = n;                    
        kernelInfo->blockz4   = 1;                    
        kernelInfo->threadx4  = 16;                   
        kernelInfo->thready4  = 16;                   
        kernelInfo->threadz4  = 1;                   
        kernelInfo->dynmicLdsSize4 = 0;
        kernelInfo->kernelPtr4= (void*)store5;  
    }
    else{
        kernelInfo->blockx1   = k / 2;                    //blockx  number
        kernelInfo->blocky1   = c / 64;                    //blocky  number
        kernelInfo->blockz1   = 1;                    //blockz  number
        kernelInfo->threadx1  = 8;                   //threadx number per block
        kernelInfo->thready1  = 16;                   //thready number per block
        kernelInfo->threadz1  = 1;                   //threadz number per block
        kernelInfo->dynmicLdsSize1 = 0;
        kernelInfo->kernelPtr1= (void*)GgGT;                 //kernel ptr

        kernelInfo->blockx2   = c;                    
        kernelInfo->blocky2   = n;                    
        kernelInfo->blockz2   = 1;                    
        kernelInfo->threadx2  = 32;                   
        kernelInfo->thready2  = 8;                   
        kernelInfo->threadz2  = 1;                   
        kernelInfo->dynmicLdsSize2 = 0;
        kernelInfo->kernelPtr2= (void*)BTdB1;

        kernelInfo->blockx3   = 16;    
        kernelInfo->blocky3   = 1;
        kernelInfo->blockz3   = 16 * n;                     
        kernelInfo->threadx3  = 16;                   
        kernelInfo->thready3  = 8;                   
        kernelInfo->threadz3  = 1;                   
        kernelInfo->dynmicLdsSize3 = 0;
        kernelInfo->kernelPtr3= (void*)matrixMul6;

        kernelInfo->blockx4   = 16;                    
        kernelInfo->blocky4   = n;                    
        kernelInfo->blockz4   = 1;                    
        kernelInfo->threadx4  = 256;
        kernelInfo->thready4  = 1;                   
        kernelInfo->threadz4  = 1;                   
        kernelInfo->dynmicLdsSize4 = 0;
        kernelInfo->kernelPtr4= (void*)store;  
    }

    pArgs->pin = problem->in;
    pArgs->pweight = problem->weight;
    pArgs->pout = problem->out;
    pArgs->transform_pin = problem->transform_pin;   
    pArgs->transform_filter = problem->transform_filter;
    pArgs->mid_result = problem->mid_result;
    pArgs->n = n;                              //batch szie             default value 1
    pArgs->c = c;                              //channel number         default value 32
    pArgs->h = h;                              //数据高                  default value 32
    pArgs->w = w;                              //数据宽                  default value 32
    pArgs->k = k;                              //卷积核数量              default value 32
    pArgs->r = r;                              //卷积核高                default value 1
    pArgs->s = s;                              //卷积核宽                default value 1
    pArgs->u = u;                              //卷积在高方向上的步长     default value 1
    pArgs->v = v;                              //卷积在宽方向上的步长     default value 1
    pArgs->p = p;                              //卷积在高方向上的补边     default value 0
    pArgs->q = q;                              //卷积在宽方向上的补边     default value 0
    pArgs->Oh = outh;
    pArgs->Ow = outw;       

    return 0;
}
