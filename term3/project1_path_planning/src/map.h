#ifndef MAP_H
#define MAP_H

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <math.h>
#include "spline.h"

class Map {

public:

  Map(){};
  Map(std::string map_file_);
  ~Map() {};

  // Convert Frenet to XY coordinates
  std::vector<double> getXY(double s, double d);

private:

  tk::spline spline_x;
  tk::spline spline_y;
  tk::spline spline_dx;
  tk::spline spline_dy;

};

#endif
