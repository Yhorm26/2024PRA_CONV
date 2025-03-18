#ifndef __CONV2D_FP16_FWD_HEADER__
#define __CONV2D_FP16_FWD_HEADER__

#define __in__
#define __out__
#define __in_out__

typedef struct
{
    _Float16*   in;                             //输入数据地址
    _Float16*   weight;                         //权值数据地址
    _Float16*   out;                            //输出数据地址
    _Float16*   transform_pin;
    _Float16*   transform_filter;
    _Float16*   mid_result;
    unsigned int      n;                              //batch szie              default value 1
    unsigned int      c;                              //channel number          default value 32
    unsigned int      h;                              //数据高                  default value 32
    unsigned int      w;                              //数据宽                  default value 32
    unsigned int      k;                              //卷积核数量              default value 32
    unsigned int      r;                              //卷积核高                default value 1
    unsigned int      s;                              //卷积核宽                default value 1
    unsigned int      u;                              //卷积在高方向上的步长     default value 1
    unsigned int      v;                              //卷积在宽方向上的步长     default value 1
    unsigned int      p;                              //卷积在高方向上的补边     default value 0
    unsigned int      q;                              //卷积在宽方向上的补边     default value 0
}problem_t;

typedef struct
{   
    unsigned int         blockx1;                    
    unsigned int         blocky1;                    
    unsigned int         blockz1;                    
    unsigned int         threadx1;                   
    unsigned int         thready1;                   
    unsigned int         threadz1;                   
    unsigned int         dynmicLdsSize1; 
    void*       kernelPtr1;  
    
    unsigned int         blockx2;                    
    unsigned int         blocky2;                    
    unsigned int         blockz2;                    
    unsigned int         threadx2;                   
    unsigned int         thready2;                   
    unsigned int         threadz2;                   
    unsigned int         dynmicLdsSize2; 
    void*       kernelPtr2;    

    unsigned int         blockx3;                    
    unsigned int         blocky3;                    
    unsigned int         blockz3;                    
    unsigned int         threadx3;                   
    unsigned int         thready3;                   
    unsigned int         threadz3;                   
    unsigned int         dynmicLdsSize3; 
    void*       kernelPtr3;        

    unsigned int         blockx4;                    
    unsigned int         blocky4;                    
    unsigned int         blockz4;                    
    unsigned int         threadx4;                   
    unsigned int         thready4;                   
    unsigned int         threadz4;                   
    unsigned int         dynmicLdsSize4; 
    void*       kernelPtr4;        
}kernelInfo_t;


int getParamsize(__in__ problem_t* problem, __out__ int* paramSize);
int getkernelInfo(__in__ problem_t* problem, __out__  kernelInfo_t* kernelInfo, __in_out__ void* param);

#endif