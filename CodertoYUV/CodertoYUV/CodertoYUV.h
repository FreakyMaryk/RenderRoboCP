# pragma once 
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


_declspec(dllexport) cudaError_t addWithCuda(unsigned char** buffer,unsigned char* ych, unsigned char* uch, unsigned char* vch, unsigned int size, int weight, int height); 