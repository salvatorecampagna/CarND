#ifndef LANE_H
#define LANE_H

#include <string>

// Highway lanes:
// LEFT, CENTER, RIGHT
// starting from the middle of the road

class Lane
{

public:

  static unsigned int get_lane(double d);
  static double get_lane_center_d(unsigned int lane);
  static double get_lane_center_d(double distance);
  static std::string lane_to_string(unsigned int lane);
  static const unsigned int LEFT_LANE = 0;
  static const unsigned int CENTER_LANE = 1;
  static const unsigned int RIGHT_LANE = 2;

};

#endif
