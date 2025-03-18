假设，以下列参数命名卷积相关参数为例，
    int      n;                              //batch szie           
    int      c;                              //channel number       
    int      h;                              //数据高               
    int      w;                              //数据宽               
    int      k;                              //卷积核数量           
    int      r;                              //卷积核高             
    int      s;                              //卷积核宽             
    int      u;                              //卷积在高方向上的步长 
    int      v;                              //卷积在宽方向上的步长 
    int      p;                              //卷积在高方向上的补边 
    int      q;                              //卷积在宽方向上的补边 
    
    int      Oh;                             //卷积在高方向上的输出大小
    int      Ow;                             //卷积在宽方向上的输出大小

有关系式：
    Oh = (int)((h-r+2*p)/u) + 1;
    Ow = (int)((w-s+2*q)/v) + 1;

则数据布局为：
    输入数据：nchw
    权值数据：kcrs
    输出数据：nkOhOw
    
    
请实现conv2d.cpp内的相关函数，完成优化；