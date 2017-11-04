#ifndef ROAD_H
#define ROAD_H

#include <string>
#include <vector>
#include <cmath>
#include "vehicle.h"
#include "lane.h"

class Road {

protected:
  std::vector<Vehicle> left_lane;
  std::vector<Vehicle> center_lane;
  std::vector<Vehicle> right_lane;

public:
  Road() {};
  ~Road() {};

  void update_lanes(std::vector<Vehicle> left_lane, std::vector<Vehicle> center_lane, std::vector<Vehicle> right_lane);
  std::vector<Vehicle> get_lane_status(LANE lane);

  LANE find_target_lane(Vehicle& car);
  bool is_lane_safe(Vehicle& car, LANE lane);
  bool free_lane(Vehicle& car, LANE lane);
};

#endif // ROAD_H
