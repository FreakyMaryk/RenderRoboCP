// Главный DLL-файл.

#include "stdafx.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <iterator>
#include <list>
#include <iostream>
#include <conio.h>
/**
 * Load a int pixel and separate it in channels
 *  Created on: 4 avr. 2012
 *   @Author: Mehdi Lauters
 *   mehdi dot lauters at gmail dot com
 */
#ifndef CUDAPIXEL_CUH_
#define CUDAPIXEL_CUH_

#if __CUDA_ARCH__ < 200 	//Compute capability 1.x architectures
#define CUPRINTF cuPrintf
#else						//Compute capability 2.x architectures
#define CUPRINTF(fmt, ...) printf("[%d, %d]:\t" fmt, \
                                                                blockIdx.y*gridDim.x+blockIdx.x,\
                                                                threadIdx.z*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x,\
                                                                __VA_ARGS__)
#endif
/**
 * availables colorspaces for one pixel
 */
typedef enum {
	RGB,//!< RGB
	YUV //!< YUV
} ColorSpaces;

/**
 * Pixel used for conversion loading from int and saving to int
 */
typedef struct {
        /**
         * The current colorspace of the pixel
         */
	ColorSpaces colorspace;

	/**
	 * the separated values of each channel
	 *
	 * in rgb space:
	 * r | g | b
	 *
	 * in yuv space
	 * y | u | v
	 */
	unsigned char channels[4];
	// r   | y
	// g   | u
	// b   | v
} CudaPixel;



/**
 * Loading and separating each channel information from a integer value which represents 4 unsigned char
 */
__device__  void pixelLoad(CudaPixel *pixel,int _value)
{
	// Dans pixel : organisation des octets = alpha | bleu | vert | rouge
	pixel->colorspace=RGB;

	// on prend que l'octet de poids faible
	pixel->channels[0] = (unsigned char)(_value & 0xff);
	//on decale d'un octet et on le recupere
	pixel->channels[1] = (unsigned char)((_value >> 8) & 0xff);
	//on redecale pour le dernier canal
	pixel->channels[2] = (unsigned char)((_value >> 16) & 0xff);


	pixel->channels[3] = (unsigned char)((_value >> 24) & 0xff);

}

/**
 * save all seperated channels of the given pixel in a integer
 */
__device__ int pixelSave(CudaPixel *pixel)
{
	int c1 = 0,
		c2 = 0,
		c3 = 0,
		c4 = 0,
		res = 0;

	c4 = pixel->channels[4];
	c4 = c4 << 24;

	// on affecte la valeur du canal et on le decale de 2 octets
	c3 = pixel->channels[2];
	c3 = c3<<16;

	//on affecte la valeur du canal et on la decale de 1 octet
	c2 = pixel->channels[1];
	c2 = c2 << 8;

	//cet octet n'est pas décalé
	c1 = pixel->channels[0];

	// or to concat the 3 octets in one int
	res = c1 | c2 | c3 | c4;
	return res;
}


/**
 * Fonction to convert one pixel from the rgb colorspace to the yuv colorspace
 * http://www.fourcc.org/fccyvrgb.php
 */
__device__ void pixelRgbToYuv(CudaPixel *pixel)
{
	pixel->colorspace = YUV;

	unsigned char r, g, b,
				  y, u, v;

	// get r g b values
	r = pixel->channels[0];
	g = pixel->channels[1];
	b = pixel->channels[2];


	/**
	 *
	 * Y  =      (0.257 * R) + (0.504 * G) + (0.098 * B) + 16
	 * Cb = U = -(0.148 * R) - (0.291 * G) + (0.439 * B) + 128
	 * Cr = V =  (0.439 * R) - (0.368 * G) - (0.071 * B) + 128
	 *
	 */



	// convert to yuv colorspace
    y = ( 0.257 * r ) + ( 0.504 * g ) + ( 0.098 * b ) + 16;

    u = - ( 0.148 * r ) - ( 0.2911 * g ) + ( 0.439 * b ) + 128;

    v = ( 0.439 * r ) - ( 0.368 * g ) - ( 0.071 * b ) + 128;


    // save values
    pixel->channels[0] = y;
    pixel->channels[1] = u;
    pixel->channels[2] = v;
}

#endif /* CUDAPIXEL_CUH_ */


/**
 * Transoding from RGBA to YV12
 *  Created on: 4 avr. 2012
 *   @Author: Mehdi Lauters
 *   mehdi dot lauters at gmail dot com
 */

/**
 * Number of division on the x axis
 */
#define X_NB_SLICES 80

/**
 * Number of division on the y axis
 */
#define Y_NB_SLICES 2

/**
 * get the linear indice of a (x,y) pixel on the frame buffer in a picture where all channels of a pixel succeed each other ( rgba rgba rgba...)
 */
__device__ int getPixelIndice(int X, int Y, int x,int y)
{
	  // computing the linear indice on the frame buffer corresponding to the x,y pixel coordinate
	  return y * X + x;
}



/**
 * kernel used to transcode all given frame from RGB to YUV
 * It was designed to process one given frame in a YUV buffer
 * @param _frameNumber the frame ID to process;
	_frameNumber argument is to say to the kernel which frame it has to process as rgb2Yv12Kernel was designed to work with a buffer of frame buffer
	myBuffer[0] = imgRgb0;
	myBuffer[1] = imgRgb1;
	myBuffer[2] = imgRgb2;
	...
	so the kernel called with _frameNumber = 1 will actually process the imgRgb1
 * @param buffer the frame array
 * @param yChannel the Y output channel
 * @param uChannel the U output channel
 * @param vChannel the V output channel
 * @param X the rgb image width
 * @param Y the rgb image height
 */

__global__ void rgb2Yv12Kernel(int _frameNumber,unsigned char* buffer,unsigned char* yChannel, unsigned char* uChannel, unsigned char* vChannel, int X, int Y)
 {

	  CudaPixel myPix;

	  // computing matricial position of the current cuda core
	  int idx = blockIdx.x * blockDim.x + threadIdx.x;
	  int idy = blockIdx.y * blockDim.y + threadIdx.y;

	  // computing X range of the current thread
	  int xMin = idx * (X / X_NB_SLICES);

	  int xMax = idx *( X / X_NB_SLICES) + (X / X_NB_SLICES);

	  // computing the Y range
	  int yMin = idy * (Y / Y_NB_SLICES);
	  int yMax = idy * (Y / Y_NB_SLICES) + (Y / Y_NB_SLICES);



	  // delta on each range
	  int xDelta = xMax - xMin;
	  int yDelta = yMax - yMin;
//
//	  if(( idy !=0))
//		  return;
//	  if(( idx >= 17))
//		  return;

	//we consider the frame buffer as a int poiter so we read one integer for each pixel (=4 unsigned char rgba)
	int* pInt = (int*) buffer; // depend on PC


	//number of pixel
	int nbPixels = X * Y;

	// the number of Y saved pixels
	int nbYPixelsSaved = 0;


	// go to Bottom left pixel
	// go back to the right x position
	// go back to the right y position
	int yBufferPixelIndice = X * Y - (X - idx * xDelta) - (yDelta * idy * X);

	// FIXME problem when idy > 3 (greens artefacts)
	int uvBufferPixelIndice = ( X/2 * Y/2 - (X/2 - idx * xDelta/2) - (yDelta/2 * idy* X/2) );



	//the indice of the pixel on the initial image
	int indice = 0;

	//int nbYuvValues=YV12_NB_PIXELS(X,Y);
	int nbYuvValues=1;


//	unsigned char *frameDest=&(bufferDest[getFrameIndice(X,Y,_frameNumber)]);

	float uMoy=0, vMoy=0;

	// browse each line
	  for(int y = yMin; y < yMax; y++)
	  {
		  //browse each pixel of the current line
		  for(int x = xMin; x < xMax; x++)
		  {
				  // computing the linear indice on the frame buffer corresponding to the x,y pixel coordinate
				  indice = getPixelIndice(X, Y,x,y);
				  if(indice > X*Y)
				  {
					  return;
				  }

                    //loading the (x,y) pixel
			        pixelLoad(&myPix,pInt[ indice ]);

			        // convert it from RGB to YUV
			        pixelRgbToYuv(&myPix);



			        //save u & v channel with subsampling
			        if( nbYPixelsSaved %2 == 0 && y%2==0)
			        {

			        	// robustness test
			        	if( ( x > 0 && x < X ) && ( y > 0 && y < Y ) )
			        	{
			        		// compute a mean of u and v around value of the subsampled pixel
			        		CudaPixel moyPix;


			        		/**
			        		 *         XXX
			        		 *         ***
			        		 *         ***
			        		 */
			        		pixelLoad(&moyPix, pInt[ getPixelIndice(X, Y, x-1 , y-1 ) ]);
			        		pixelRgbToYuv(&moyPix);
			        		uMoy += moyPix.channels[1];
			        		vMoy += moyPix.channels[2];

			        		pixelLoad(&moyPix, pInt[ getPixelIndice(X, Y, x , y-1 ) ]);
			        		pixelRgbToYuv(&moyPix);
			        		uMoy += moyPix.channels[1];
			        		vMoy += moyPix.channels[2];

			        		pixelLoad(&moyPix, pInt[ getPixelIndice(X, Y, x+1 , y-1) ]);
			        		pixelRgbToYuv(&moyPix);
			        		uMoy += moyPix.channels[1];
			        		vMoy += moyPix.channels[2];


			        		/**
                                                 *         ***
                                                 *         XXX
                                                 *         ***
                                                 */
			        		pixelLoad(&moyPix, pInt[ getPixelIndice(X, Y, x-1 , y ) ]);
			        		pixelRgbToYuv(&moyPix);
			        		uMoy += moyPix.channels[1];
			        		vMoy += moyPix.channels[2];

			        		pixelLoad(&moyPix, pInt[ getPixelIndice(X, Y, x , y ) ]);
			        		pixelRgbToYuv(&moyPix);
			        		uMoy += moyPix.channels[1];
			        		vMoy += moyPix.channels[2];

			        		pixelLoad(&moyPix, pInt[ getPixelIndice(X, Y, x+1 , y ) ]);
			        		pixelRgbToYuv(&moyPix);
			        		uMoy += moyPix.channels[1];
			        		vMoy += moyPix.channels[2];


			        		/**
                                                 *         ***
                                                 *         ***
                                                 *         XXX
                                                 */
			        		pixelLoad(&moyPix, pInt[ getPixelIndice(X, Y, x-1 , y+1 ) ]);
			        		pixelRgbToYuv(&moyPix);
			        		uMoy += moyPix.channels[1];
			        		vMoy += moyPix.channels[2];

			        		pixelLoad(&moyPix, pInt[ getPixelIndice(X, Y, x , y+1) ]);
			        		pixelRgbToYuv(&moyPix);
			        		uMoy += moyPix.channels[1];
			        		vMoy += moyPix.channels[2];

			        		pixelLoad(&moyPix, pInt[ getPixelIndice(X, Y, x+1 , y+1 ) ]);
			        		pixelRgbToYuv(&moyPix);
			        		uMoy += moyPix.channels[1];
			        		vMoy += moyPix.channels[2];





			        	}
			        	else
			        	{
			        		vMoy=9*255; // set it to white (debug)
			        	}

			        	int uIndice = uvBufferPixelIndice;
			        	int vIndice = uvBufferPixelIndice;


			        	uChannel[ uIndice ] = myPix.channels[1];//floor( uMoy/9);
			        	vChannel[ vIndice ] = myPix.channels[2];//floor( vMoy/9);
                                        uMoy=0;
                                        vMoy=0;
                                        uvBufferPixelIndice++;
			        }

			      //y channel
			        yChannel[ yBufferPixelIndice ] =myPix.channels[ 0 ];

                              // increment the saved pixels number
                              nbYPixelsSaved++;

                              // go to the next pixel of the current line of the y buffer
                              yBufferPixelIndice++;


			  }
			  // go one line up on the buffer
			  yBufferPixelIndice = yBufferPixelIndice - ( xDelta + X );

			  uvBufferPixelIndice = uvBufferPixelIndice - ( xDelta/4 + X/4 );
		  }
	  
	  }

// Helper function for using CUDA to add vectors in parallel.
_declspec(dllexport) cudaError_t addWithCuda(unsigned char* buffer,unsigned char* ych, unsigned char* uch, unsigned char* vch, unsigned int size, int width, int height)
{
	dim3 threads(X_NB_SLICES, Y_NB_SLICES);
	dim3 block(1,1);
	int size2 = size*sizeof(int);
	unsigned char * dev_buff = new unsigned char[size*sizeof(int)];
	unsigned char * dev_uch = new unsigned char[size];
	unsigned char * dev_vch = new unsigned char[size];
	unsigned char * dev_ych = new unsigned char[size];
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_ych, size * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_uch, size * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_vch, size  * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_buff, size *  sizeof(int)*2);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_buff, buffer, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
	rgb2Yv12Kernel<<< block, threads >>> (0, dev_buff, dev_ych, dev_uch, dev_vch, width, height);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
     cudaStatus = cudaDeviceSynchronize();
    while (cudaStatus != cudaSuccess) {
		cudaStatus = cudaDeviceSynchronize();
        }
	
    // Copy output vector from GPU buffer to host memory
    cudaStatus = cudaMemcpy(ych, dev_ych, size * sizeof(unsigned char ), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(uch, dev_uch, size * sizeof(unsigned char ), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(vch, dev_vch, size * sizeof(unsigned char ), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_ych);
    cudaFree(dev_uch);
    cudaFree(dev_vch);
    
    return cudaStatus;
}




