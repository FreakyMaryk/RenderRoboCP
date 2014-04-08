#include <vtkVersion.h>
#if VTK_MAJOR_VERSION == 6
int main(int, char *argv[])
{
  std::cout << argv[0] << " requires VTK 5.10 or earlier. This VTK version is " << vtkVersion::GetVTKVersion() << std::endl;
  return EXIT_SUCCESS;
}
#else
// This demo creates depth map for a polydata instance by extracting
// exact ZBuffer values.
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vtkSmartPointer.h> 
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkPLYReader.h>
#include <vtkBMPWriter.h>
#include <vtkWindowToImageFilter.h>
#include <vtkImageShiftScale.h>
#include <vtkImageData.h>
#include <vtkSmartPointer.h>
#include <vtkVersion.h>
#include <list>

/*
* This function realized in .dll. It calls coder to YUV from ARGB
* input params: buffer - array of ints(A - 8 bit, R - 8 bit, G - 8 bit, B - 8 bit) 
* ych, uch, vch - arrays for new colorspace coordinates 
* size - framesize
* width, height - picture size
*/

cudaError_t addWithCuda(unsigned char* buffer,unsigned char* ych, unsigned char* uch, unsigned char* vch, unsigned int size, int weight, int height);

/*
* This function in realized in .dll. It calls writer from three arrays with YUV colorspace coordinates.
* input params: path for output file
* X, Y - width,height
* yChannel, uChannel, vChannel - input arrays with YUV data
*/

void writeYUVframe(std::string filename, int _screenId, int X, int Y, unsigned char *yChannel, unsigned char *uChannel, unsigned char *vChannel);

/*
*this struct for writing unsigned char information in int
*/

typedef struct {
	unsigned char channel : 8;
	unsigned char r : 8;
	unsigned char g : 8;
	unsigned char b : 8;
	} Pix_struct;

/*
* input args sets in params of progect: Project->Configuration properties->Debug->Argument of cmd
*  first - .ply file, sechond - .bmp file 
*/

int main(int argc, char *argv[])
{
	if (argc < 3)
    {
      std::cout << "Usage: " << argv[0]
                << " input(.ply) output(.bmp)"
                << std::endl;
      return EXIT_FAILURE;
    }
  // Set VTK enviroment
  vtkSmartPointer<vtkPolyDataMapper> mapper =
    vtkSmartPointer<vtkPolyDataMapper>::New();
  vtkSmartPointer<vtkActor> actor =
    vtkSmartPointer<vtkActor>::New();
  vtkSmartPointer<vtkRenderer> renderer =
    vtkSmartPointer<vtkRenderer>::New();
  vtkSmartPointer<vtkRenderWindow> renWin =
    vtkSmartPointer<vtkRenderWindow>::New();
  vtkSmartPointer<vtkRenderWindowInteractor> interactor =
    vtkSmartPointer<vtkRenderWindowInteractor>::New();
  vtkSmartPointer<vtkPLYReader> fileReader =
    vtkSmartPointer<vtkPLYReader>::New();

  vtkSmartPointer<vtkWindowToImageFilter> filter =
    vtkSmartPointer<vtkWindowToImageFilter>::New();
 
  vtkSmartPointer<vtkBMPWriter> imageWriter =
    vtkSmartPointer<vtkBMPWriter>::New();
  
  vtkSmartPointer<vtkImageShiftScale> scale =
    vtkSmartPointer<vtkImageShiftScale>::New();
 
  // Read .ply file
  fileReader->SetFileName(argv[1]);
 
  //Build visualization enviroment
  mapper->SetInputConnection(fileReader->GetOutputPort());
  actor->SetMapper(mapper);
  renderer->AddActor(actor);
  renWin->AddRenderer(renderer);
  renWin->SetOffScreenRendering(1);
  renWin->SetSize(720,480);
  interactor->SetRenderWindow(renWin);
  renWin->Render();
 
  // Create Depth Map
  filter->SetInput(renWin);
 // filter->SetInputBufferTypeToZBuffer();
  filter->SetInputBufferTypeToRGB();
 
  scale->SetOutputScalarTypeToUnsignedChar();
  scale->SetInputConnection(filter->GetOutputPort());
  scale->SetShift(0);
  scale->SetScale(-255);
 
  filter->Update();
  vtkImageData * id = filter->GetOutput();
  id->Update();
  imageWriter->SetInput(id);
  imageWriter->SetFileName(argv[2]);
  imageWriter->SetInputConnection(scale->GetOutputPort());
  
  // Get frame size
  int width = id->GetDimensions()[0];
  int height = id->GetDimensions()[1];
  int frameSize = height*width;
 
  // Get pixel data in list of values
  int tmp;
  std::list <int> values;
  for (int y = 0; y < height; y++)
    {
    for (int x = 0; x < width; x++)
      {
	  unsigned char* pixel = static_cast<unsigned char*>(id->GetScalarPointer(x,y,0));
	   (* (Pix_struct *)(&tmp)).channel = 0; 
		(* (Pix_struct *)(&tmp)).r = pixel[0];
		(* (Pix_struct *)(&tmp)).g = pixel[1];
		(* (Pix_struct *)(&tmp)).b = pixel[2];
		values.push_front(tmp);
	}
    }
    // Write list of int in a char array
	unsigned char ** buff = new unsigned char * [1];
	buff[0] = new unsigned char[width*height*sizeof(int)];
	unsigned char * uch = new unsigned char[frameSize];
	unsigned char * vch = new unsigned char[frameSize];
	unsigned char * ych = new unsigned char[frameSize];
	std::list<int>::iterator it = values.begin();
	int * kk;
	for (int j = 0; j < frameSize; j++, it++) {
	kk = (int*) ( (int*)(buff[0] + j*sizeof(int)));
	*kk = *it;
	}

	// Start Cuda coder from ARGB to YUV
	 cudaError_t cudaStatus = addWithCuda(buff[0],ych,uch,vch,frameSize, width, height);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
    }

	// Write image in .BMP
   imageWriter->Write();
   // Start writing in .yuv file
   writeYUVframe("C:\\Users\\FreakyMaryK\\Desktop\\SMR", 60, width, height, ych, uch, vch);
  return EXIT_SUCCESS;
}
#endif
