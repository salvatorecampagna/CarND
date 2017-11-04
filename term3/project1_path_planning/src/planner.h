#ifndef PLANNER_H
#define PLANNER_H

#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "Eigen-3.3/Eigen/Dense"
#include <vector>

#include "map.h"
#include "road.h"
#include "state.h"
#include "lane.h"

class Planner {

public:

  Planner();
  ~Planner(){};

  std::vector<double> jerk_min_trajectory(std::vector<double> start, std::vector<double> end, double T);
  void compute_trajectory(Map& map, std::vector<std::vector<double>>& trajectory);
  void new_trajectory(Map& map, Road& road, Vehicle& car, std::vector<std::vector<double>>& trajectory);
  void start(Vehicle& car);
  void keep_lane(Vehicle& car);
  void slow_down(Vehicle& car);
  void change_lane(Vehicle& car, LANE target_lane);
  void update_state(LANE current_lane, LANE target_lane);

private:

  int n;
  STATE state;
  std::vector<double> start_s;
  std::vector<double> end_s;
  std::vector<double> start_d;
  std::vector<double> end_d;
  bool recompute_trajectory;
};

#endif
