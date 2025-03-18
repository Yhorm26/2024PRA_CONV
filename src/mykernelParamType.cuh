#ifndef MYKERNELPARAMTYPE_H
#define MYKERNELPARAMTYPE_H


typedef struct mykernelParamType
{
    _Float16*  __restrict__ pin;                            //输入数据地址
    _Float16*  __restrict__ pweight;                        //权值数据地址
    _Float16*  __restrict__ pout;                           //输出数据地址
    _Float16*  __restrict__ transform_pin;     
    _Float16*  __restrict__ transform_filter;  
    _Float16*  __restrict__ mid_result;     
    short      n;                              //batch szie            
    short      c;                              //channel number        
    short      h;                              //数据高                
    short      w;                              //数据宽                
    short      k;                              //卷积核数量            
    short      r;                              //卷积核高              
    short      s;                              //卷积核宽              
    short      u;                              //卷积在高方向上的步长  
    short      v;                              //卷积在宽方向上的步长  
    short      p;                              //卷积在高方向上的补边  
    short      q;                              //卷积在宽方向上的补边  
    short      Oh;                             //卷积在高方向上输出大小    
    short      Ow;                             //卷积在宽方向上输出大小
}mykernelParamType;
#endif