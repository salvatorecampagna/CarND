#include "map.h"

Map::Map(std::string map_file_)
{
  std::vector<double> map_waypoints_x;
  std::vector<double> map_waypoints_y;
  std::vector<double> map_waypoints_s;
  std::vector<double> map_waypoints_dx;
  std::vector<double> map_waypoints_dy;
  std::string line;

  // Open map to read waypoints
  std::ifstream in_map(map_file_.c_str(), std::ifstream::in);

  while (getline(in_map, line))
  {
    std::istringstream iss(line);
    double x;
    double y;
    float s;
    float dx;
    float dy;
    iss >> x;
    iss >> y;
    iss >> s;
    iss >> dx;
    iss >> dy;
    map_waypoints_x.push_back(x);
    map_waypoints_y.push_back(y);
    map_waypoints_s.push_back(s);
    map_waypoints_dx.push_back(dx);
    map_waypoints_dy.push_back(dy);
  }

  in_map.close();

  // Set waypoints for each spline
  this->spline_x.set_points(map_waypoints_s, map_waypoints_x);
  this->spline_y.set_points(map_waypoints_s, map_waypoints_y);
  this->spline_dx.set_points(map_waypoints_s, map_waypoints_dx);
  this->spline_dy.set_points(map_waypoints_s, map_waypoints_dy);

}

std::vector<double> Map::getXY(double s, double d)
{
  std::vector<double> xy;
  double waypoints_x, waypoints_y, waypoints_dx, waypoints_dy, next_x, next_y;

  waypoints_x = this->spline_x(s);
  waypoints_y = this->spline_y(s);
  waypoints_dx = this->spline_dx(s);
  waypoints_dy = this->spline_dy(s);

  next_x = waypoints_x + waypoints_dx * d;
  next_y = waypoints_y + waypoints_dy * d;

  xy.push_back(next_x);
  xy.push_back(next_y);

  return xy;
}
