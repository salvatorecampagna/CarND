#ifndef VEHICLE_H
#define VEHICLE_H

#include <vector>
#include "lane.h"

class Vehicle {

public:

  Vehicle();
  Vehicle(int id, double x, double y, double v, double s, double d);
  ~Vehicle(){};

  int get_id();
  double get_x();
  double get_y();
  double get_v();
  double get_s();
  double get_d();
  double get_yaw();
  unsigned int lane();
  void update_vehicle_status(double x, double y, double v, double s, double d, double yaw);
  void set_previous_s(std::vector<double> previous_s);
  void set_previous_d(std::vector<double> previous_d);
  std::vector<double> get_previous_s();
  std::vector<double> get_previous_d();

private:

  int id;
  double x;
  double y;
  double v;
  double s;
  double d;
  double yaw;

  std::vector<double> previous_s;
  std::vector<double> previous_d;

};

#endif
