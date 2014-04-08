# pragma once
#include <string>
_declspec(dllexport) void writeYUVframe(std::string filename, int _screenId, int X, int Y, unsigned char *yChannel, unsigned char *uChannel, unsigned char *vChannel)