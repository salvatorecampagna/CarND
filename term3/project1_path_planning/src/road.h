#ifndef ROAD_H
#define ROAD_H

#include <string>
#include <vector>
#include <cmath>
#include "vehicle.h"
#include "lane.h"

class Road {

public:

  Road() {};
  ~Road() {};

  void update_lanes(std::vector<Vehicle> left_lane, std::vector<Vehicle> center_lane, std::vector<Vehicle> right_lane);
  std::vector<Vehicle> get_lane_status(unsigned int lane);
  unsigned int find_target_lane(Vehicle& car);
  bool is_lane_safe(Vehicle& car, unsigned int lane);
  bool is_lane_free(Vehicle& car, unsigned int lane);
  unsigned int get_cars_per_lane(unsigned int lane);
  unsigned int get_total_cars();
  double get_lane_avg_speed(unsigned int lane);
  double get_front_vehicle_speed(Vehicle& car);
  double get_front_vehicle_speed(Vehicle& car, unsigned int lane);
  double get_front_vehicle_distance(Vehicle& car);

private:

  double lane_cars_cost(unsigned int lane);
  double lane_speed_cost(unsigned int lane);
  double lane_distance_cost(Vehicle& car, unsigned int lane);
  double lane_cost(Vehicle& car, unsigned int lane);
  unsigned int find_low_cost_lane(Vehicle& car);

  unsigned int total_cars;
  unsigned int cars_per_lane[3];
  double avg_speed_per_lane[3];
  std::vector<Vehicle> left_lane;
  std::vector<Vehicle> center_lane;
  std::vector<Vehicle> right_lane;
};

#endif // ROAD_H
