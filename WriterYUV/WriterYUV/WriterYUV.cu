/**
 * Writting an yuv or y4m file readable from mplayer and so one
 *  Created on: 4 avr. 2012
 *   @Author: Mehdi Lauters
 *   mehdi dot lauters at gmail dot com
 */
#include <string>

static FILE * yuvFiles= NULL;

/**
 * write a yuv video file in yuv or y4m format in the filename_XxY.ext file
 * y4m format is usefull because it can be read in more player easily
 * yuv format is the input format of h264 video encoder
 */
_declspec(dllexport) void writeYUVframe(std::string filename, int _screenId, int X, int Y, unsigned char *yChannel, unsigned char *uChannel, unsigned char *vChannel)
{
//	mplayer -demuxer rawvideo -rawvideo w=X:h=Y:format=yv12 filename.yuv

	/**
	 * File in which the video will be saved
	 */
        char buffer[255];
		int frameSize = X*Y;
        sprintf(buffer, "%s\\screen_%d_%dx%d_.yuv",filename.c_str(), _screenId, X, Y);

        // if the file as to be created

		if( yuvFiles == NULL )
	{
		yuvFiles = fopen( buffer, "w");

	}
	else
	{
	  // we just open it
		yuvFiles=fopen( buffer, "a+");
	}


	// write the frame data	
		bool f= 0;
		int k = 0;
			for(int i = 0; i < frameSize; i++) 
			{
					fprintf(yuvFiles,"%c",yChannel[i]);
			}
		    for (int j = 0; j < Y; j++ ){ 
			if(j%2) { k = j * X; }
			else k = (j-1) * X;
			for(int i=j*X; i < (j+1)*X; i++) {
			if (!f) {
						fprintf(yuvFiles,"%c",uChannel[k]);
						k++;
						f = 1;
			}
			else {

						fprintf(yuvFiles,"%c",vChannel[k]);
						k++;
						f = 0;
				 }
			k++;
			}
		}
        // close the file
	fclose(yuvFiles);
}
