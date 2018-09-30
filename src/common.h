#include <sys/time.h>
// #include <stdio.h>

#ifndef _COMMON_H
#define _COMMON_H

inline double cpuSecond()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}


#endif
