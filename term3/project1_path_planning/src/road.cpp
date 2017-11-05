#include "road.h"
#include "params.h"
#include <iostream>

void Road::update_lanes(std::vector<Vehicle> left_lane, std::vector<Vehicle> center_lane, std::vector<Vehicle> right_lane)
{
  double speed_accumulator;
  this->left_lane = left_lane;
  this->center_lane = center_lane;
  this->right_lane = right_lane;
  this->cars_per_lane[Lane::LEFT_LANE] = left_lane.size();
  this->cars_per_lane[Lane::CENTER_LANE] = center_lane.size();
  this->cars_per_lane[Lane::RIGHT_LANE] = right_lane.size();
  this->total_cars = left_lane.size() + center_lane.size() + right_lane.size();

  speed_accumulator = 0;
  for (auto i = 0; i < left_lane.size(); i++)
  {
    speed_accumulator += left_lane[i].get_v();
  }
  if (left_lane.size() > 0)
  {
    this->avg_speed_per_lane[Lane::LEFT_LANE] = speed_accumulator / left_lane.size();
  }
  else
  {
    this->avg_speed_per_lane[Lane::LEFT_LANE] = SPEED_LIMIT;
  }

  speed_accumulator = 0;
  for (auto i = 0; i < center_lane.size(); i++)
  {
    speed_accumulator += center_lane[i].get_v();
  }
  if (center_lane.size() > 0)
  {
    this->avg_speed_per_lane[Lane::CENTER_LANE] = speed_accumulator / center_lane.size();
  }
  else
  {
    this->avg_speed_per_lane[Lane::CENTER_LANE] = SPEED_LIMIT;
  }

  speed_accumulator = 0;
  for (auto i = 0; i < right_lane.size(); i++)
  {
    speed_accumulator += right_lane[i].get_v();
  }
  if (right_lane.size() > 0)
  {
    this->avg_speed_per_lane[Lane::RIGHT_LANE] = speed_accumulator / right_lane.size();
  }
  else
  {
    this->avg_speed_per_lane[Lane::RIGHT_LANE] = SPEED_LIMIT;
  }
}

std::vector<Vehicle> Road::get_lane_status(unsigned int lane)
{
  std::vector<Vehicle> vehicles_in_lane;

  if (lane == Lane::LEFT_LANE)
  {
    vehicles_in_lane = this->left_lane;
  }
  else if (lane == Lane::CENTER_LANE)
  {
    vehicles_in_lane = this->center_lane;
  }
  else
  {
    vehicles_in_lane = this->right_lane;
  }

  return vehicles_in_lane;
}

bool Road::is_lane_safe(Vehicle& car, unsigned int lane)
{
  std::vector<Vehicle> vehicles_in_lane = this->get_lane_status(lane);
  bool safe = true;

  // Check if there is a vehicle that is too close
  // distance from car < BUFFER_DISTANCE
  for (int i = 0; i < vehicles_in_lane.size(); i++)
  {
    double distance = vehicles_in_lane[i].get_s() - car.get_s();
    if(distance > 0 && distance < BUFFER_DISTANCE)
    {
      // Vehicle too close
      safe = false;
    }
  }

  return safe;
}

bool Road::is_lane_free(Vehicle& car, unsigned int lane)
{
  std::vector<Vehicle> target_lane = this->get_lane_status(lane);
  bool is_lane_free = true;

  // Check if there is enough space in the target lane
  // enough space => distance < COLLISION_DISTANCE
  for (int i = 0; i < target_lane.size(); i++)
  {
    double distance = std::abs(target_lane[i].get_s() - car.get_s());
    if(distance < COLLISION_DISTANCE)
    {
      is_lane_free = false;
    }
  }

  return is_lane_free;
}

unsigned int Road::find_low_cost_lane(Vehicle& car)
{
  double left_lane_cost = lane_cost(car, Lane::LEFT_LANE);
  double center_lane_cost = lane_cost(car, Lane::CENTER_LANE);
  double right_lane_cost = lane_cost(car, Lane::RIGHT_LANE);
  double min_cost = left_lane_cost;
  unsigned int best_lane = Lane::LEFT_LANE;

  if (center_lane_cost < min_cost)
  {
    min_cost = center_lane_cost;
    best_lane = Lane::CENTER_LANE;
  }
  if (right_lane_cost < min_cost)
  {
    min_cost = right_lane_cost;
    best_lane = Lane::RIGHT_LANE;
  }

  //std::cout << left_lane_cost << " " << center_lane_cost << " " << right_lane_cost << std::endl;
  return best_lane;
}

unsigned int Road::find_target_lane(Vehicle& car)
{
  unsigned int car_lane = car.lane();
  unsigned int target_lane = car_lane;

  unsigned int best_lane = find_low_cost_lane(car);
  //bool best_lane_free = this->is_lane_free(car, best_lane);
  //std::string best_lane_s = Lane::lane_to_string(best_lane);
  //std::string best_lane_free_s = best_lane_free ? "true" : "false";
  //std::cout << "Best lane: " << best_lane_s << " " << best_lane_free_s << std::endl;

  // If the best lane is the opposite one go through the center lane first
  if (car_lane == Lane::LEFT_LANE && best_lane == Lane::RIGHT_LANE)
  {
    if(this->is_lane_free(car, Lane::CENTER_LANE))
    {
      std::cout << "[left lane => center lane => right lane]" << std::endl;
      target_lane = Lane::CENTER_LANE;
    }
  }
  else if (car_lane == Lane::RIGHT_LANE && best_lane == Lane::LEFT_LANE)
  {
    if(this->is_lane_free(car, Lane::CENTER_LANE))
    {
      std::cout << "[right lane => center lane => left lane]" << std::endl;
      target_lane = Lane::CENTER_LANE;
    }
  }
  // If we are in the left or right lane check if we can move to
  // center lane
  else if (car_lane == Lane::LEFT_LANE || car_lane == Lane::RIGHT_LANE)
  {
    if (best_lane == Lane::CENTER_LANE && this->is_lane_free(car, Lane::CENTER_LANE))
    {
      target_lane = Lane::CENTER_LANE;
    }
  }
  // If we are in the center lane check if we can move to the
  // left or right lane
  else
  {
    if (best_lane == Lane::LEFT_LANE && this->is_lane_free(car, Lane::LEFT_LANE))
    {
      target_lane = Lane::LEFT_LANE;
    }
    else if (best_lane == Lane::RIGHT_LANE && this->is_lane_free(car, Lane::RIGHT_LANE))
    {
      target_lane = Lane::RIGHT_LANE;
    }
  }

  return target_lane;
}

unsigned int Road::get_cars_per_lane(unsigned int lane)
{
  return this->cars_per_lane[lane];
}

unsigned int Road::get_total_cars()
{
  return this->total_cars;
}

double Road::get_lane_avg_speed(unsigned int lane)
{
  return this->avg_speed_per_lane[lane];
}

double Road::get_front_vehicle_speed(Vehicle& car)
{
  std::vector<Vehicle> vehicle_lane = this->get_lane_status(car.lane());
  double min_distance = 1e6;
  double speed = SPEED_LIMIT;

  for (int i = 0; i < vehicle_lane.size(); i++)
  {
    double distance = vehicle_lane[i].get_s() - car.get_s();
    if(distance > 0 && distance < min_distance)
    {
        speed = vehicle_lane[i].get_v();
        min_distance = distance;
    }
  }

  return min_distance > BUFFER_DISTANCE ? SPEED_LIMIT : speed;
}

double Road::get_front_vehicle_distance(Vehicle& car)
{
  std::vector<Vehicle> vehicle_lane = this->get_lane_status(car.lane());
  double min_distance = 1e6;

  for (int i = 0; i < vehicle_lane.size(); i++)
  {
    double distance = vehicle_lane[i].get_s() - car.get_s();
    if(distance > 0 && distance < min_distance)
    {
        min_distance = distance;
    }
  }

  return min_distance;
}

double Road::get_front_vehicle_speed(Vehicle& car, unsigned int lane)
{
  std::vector<Vehicle> vehicles_in_lane = this->get_lane_status(lane);
  double min_distance = 1e6;
  double speed = SPEED_LIMIT;

  for (int i = 0; i < vehicles_in_lane.size(); i++)
  {
    double distance = vehicles_in_lane[i].get_s() - car.get_s();
    if(distance > 0 && distance < min_distance)
    {
        speed = vehicles_in_lane[i].get_v();
        min_distance = distance;
    }
  }

  return min_distance > BUFFER_DISTANCE ? SPEED_LIMIT : speed;
}

double Road::lane_cars_cost(unsigned int lane)
{
  return 1.0  * this->get_cars_per_lane(lane) / this->get_total_cars();
}

double Road::lane_speed_cost(unsigned int lane)
{
  return std::max(0.0, 1.0 -  this->get_lane_avg_speed(lane) / SPEED_LIMIT);
}

// Find the lane whose closer vehicles is further
double Road::lane_distance_cost(Vehicle& car, unsigned int lane)
{
  std::vector<Vehicle> vehicles_in_lane = this->get_lane_status(lane);
  double min_distance = 1e6;

  for (int i = 0; i < vehicles_in_lane.size(); i++)
  {
    double distance = vehicles_in_lane[i].get_s() - car.get_s();
    if(distance > 0 && distance < min_distance)
    {
      min_distance = distance;
    }
  }

  return 1.0 / min_distance;
}

double Road::lane_cost(Vehicle& car, unsigned int lane)
{
  double cars_cost = CARS_PER_LANE_COST_WEIGHT * lane_cars_cost(lane);
  double speed_cost = LANE_SPEED_COST_WEIGHT * lane_speed_cost(lane);
  double distance_cost = LANE_DISTANCE_COST_WEIGHT * lane_distance_cost(car, lane);

  // std::cout << "Lane: " << Lane::lane_to_string(lane) << std::endl;
  // std::cout << "cars cost: " << cars_cost << std::endl;
  // std::cout << "speed cost: " << speed_cost << std::endl;
  // std::cout << "distance cost: " << distance_cost << std::endl;

  return cars_cost + speed_cost + distance_cost;
}
