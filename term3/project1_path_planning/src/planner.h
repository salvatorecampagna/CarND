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
  void start(Road& road, Vehicle& car);
  void keep_lane(Road& road, Vehicle& car);
  void slow_down(Road& road, Vehicle& car);
  void change_lane(Road& road, Vehicle& car, unsigned int target_lane);
  void update_state(unsigned int current_lane, unsigned int target_lane);

private:
  double acceleration(Vehicle& car, double distance, double speed);
  double brake(Vehicle& car, double distance, double speed);
  double overtake(Vehicle& car, double distance, double curr_lane_speed, double target_lane_speed);

  int n;
  STATE state;
  std::vector<double> start_s;
  std::vector<double> end_s;
  std::vector<double> start_d;
  std::vector<double> end_d;
  bool recompute_trajectory;
};

#endif
