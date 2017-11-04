#include "road.h"

const double COLLISION_DISTANCE = 25.0; // meters
const double BUFFER_DISTANCE = 30.0; // meters

void Road::update_lanes(std::vector<Vehicle> left_lane, std::vector<Vehicle> center_lane, std::vector<Vehicle> right_lane)
{
  this->left_lane = left_lane;
  this->center_lane = center_lane;
  this->right_lane = right_lane;
}

std::vector<Vehicle> Road::get_lane_status(LANE lane)
{
  std::vector<Vehicle> vehicles_in_lane;

  if (lane == LANE::LEFT_LANE)
  {
    vehicles_in_lane = this->left_lane;
  }
  else if (lane == LANE::CENTER_LANE)
  {
    vehicles_in_lane = this->center_lane;
  }
  else
  {
    vehicles_in_lane = this->right_lane;
  }

  return vehicles_in_lane;
}

bool Road::is_lane_safe(Vehicle& car, LANE lane)
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

bool Road::free_lane(Vehicle& car, LANE lane)
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

LANE Road::find_target_lane(Vehicle& car)
{
  LANE car_lane = car.lane();
  LANE target_lane = car_lane;

  // If we are in the left or right lane check if we can move to
  // center lane
  if (car_lane == LANE::LEFT_LANE || car_lane == LANE::RIGHT_LANE)
  {
    if (this->free_lane(car, LANE::CENTER_LANE))
    {
      target_lane = LANE::CENTER_LANE;
    }
  }
  // If we are in the center lane check if we can move to the
  // left or right lane
  else
  {
    if (this->free_lane(car, LANE::LEFT_LANE))
    {
      target_lane = LANE::LEFT_LANE;
    }
    else if (this->free_lane(car, LANE::RIGHT_LANE))
    {
      target_lane = LANE::RIGHT_LANE;
    }
  }

  return target_lane;
}
