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
 
int main(int argc, char *argv[])
{
  if (argc < 3)
    {
      std::cout << "Usage: " << argv[0]
                << " input(.ply) output(.bmp)"
                << std::endl;
      return EXIT_FAILURE;
    }
 
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
  interactor->SetRenderWindow(renWin);
  renWin->Render();
 
  // Create Depth Map
  filter->SetInput(renWin);
  filter->SetMagnification(3);
 // filter->SetInputBufferTypeToZBuffer();
  filter->SetInputBufferTypeToRGB();
 
  scale->SetOutputScalarTypeToUnsignedChar();
  scale->SetInputConnection(filter->GetOutputPort());
  scale->SetShift(0);
  scale->SetScale(-255);
 
  // Get pixel data
  imageWriter->SetInput(filter->GetOutput());
  filter->Update();
  vtkImageData * id = filter->GetOutput();
  id->Update();
  unsigned char* pPix;
  pPix = (unsigned char*)id->GetScalarPointer();

  for(int ee=0;ee<1041;ee++){
    int pp = pPix[ee];
    cout<<" "<<pp;
  }
 
  return EXIT_SUCCESS;
}
#endif

//#include <vtkVersion.h>
//#include <vtkSmartPointer.h>
//#include <vtkImageData.h>
// 
//int main(int, char *[])
//{
//  // Create an image data
//  vtkSmartPointer<vtkImageData> imageData = 
//    vtkSmartPointer<vtkImageData>::New();
// 
//  // Specify the size of the image data
//  imageData->SetDimensions(2,3,1);
//#if VTK_MAJOR_VERSION <= 5
//  imageData->SetNumberOfScalarComponents(1);
//  imageData->SetScalarTypeToDouble();
//#else
//  imageData->AllocateScalars(VTK_DOUBLE,1);
//#endif
// 
//  int* dims = imageData->GetDimensions();
//  // int dims[3]; // can't do this
// 
//  std::cout << "Dims: " << " x: " << dims[0] << " y: " << dims[1] << " z: " << dims[2] << std::endl;
// 
//  std::cout << "Number of points: " << imageData->GetNumberOfPoints() << std::endl;
//  std::cout << "Number of cells: " << imageData->GetNumberOfCells() << std::endl;
// 
//  // Fill every entry of the image data with "2.0"
//  for (int z = 0; z < dims[2]; z++)
//    {
//    for (int y = 0; y < dims[1]; y++)
//      {
//      for (int x = 0; x < dims[0]; x++)
//        {
//        double* pixel = static_cast<double*>(imageData->GetScalarPointer(x,y,z));
//        pixel[0] = 2.0;
//        }
//      }
//    }
// 
//  // Retrieve the entries from the image data and print them to the screen
//  for (int z = 0; z < dims[2]; z++)
//    {
//    for (int y = 0; y < dims[1]; y++)
//      {
//      for (int x = 0; x < dims[0]; x++)
//        {
//        double* pixel = static_cast<double*>(imageData->GetScalarPointer(x,y,z));
//        // do something with v
//        std::cout << pixel[0] << " ";
//        }
//      std::cout << std::endl;
//      }
//    std::cout << std::endl;
//    }
//