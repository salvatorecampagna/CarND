#include "lane.h"
#include <string>

// Convert a lane to string
std::string Lane::lane_to_string(unsigned int lane)
{
  std::string lane_string;

  switch(lane)
  {
    case Lane::LEFT_LANE:
      lane_string = "left lane";
      break;

    case Lane::RIGHT_LANE:
      lane_string = "right lane";
      break;

    case Lane::CENTER_LANE:
      lane_string = "center lane";
      break;

    default:
      lane_string = "unknown";
  }

  return lane_string;
}

// Get the lane given the distance from center of the road
unsigned int Lane::get_lane(double d)
{
  unsigned int lane;

  // Left lane 0 - 4 meters
  if (d < 4.0)
  {
    lane = Lane::LEFT_LANE;
  }
  // Center lane 4 - 8 meters
  else if ((d >= 4.0) && (d < 8.0))
  {
    lane = Lane::CENTER_LANE;
  }
  // Right lane 8 - 12 meters
  else
  {
    lane = Lane::RIGHT_LANE;
  }

  return lane;
}

// Get the lane center given the lane
double Lane::get_lane_center_d(unsigned int lane)
{
  double d;

  switch (lane)
  {
    case Lane::LEFT_LANE:
      d = 2.0; // Left lane center
      break;

    case Lane::CENTER_LANE:
      d = 6.0; // Center lane center
      break;

    case Lane::RIGHT_LANE:
      d = 10.0; // Right lane ceter
      break;

    default:
      d = 6.0; // Center lane center
  }

  return d;
}

// Get the lane ceter given the distance
double Lane::get_lane_center_d(double distance){
  double d;

  if (distance < 4.0)
  {
    d = 2.0; // left lane center
  }
  else if ((distance >= 4.0) && (distance < 8.0))
  {
    d = 6.0; // center lane center
  }
  else
  {
    d = 10.0; // right lane center
  }

  return d;
}
