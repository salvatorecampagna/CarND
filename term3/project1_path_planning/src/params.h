#ifndef PARAMS_H
#define PARAMS_H

const double MAX_D = 11.90; // meters
const double MAX_S = 6914.14925765991; // meters
const double SPEED_LIMIT = 20.5; // meters / second
const double DELTA_T = 0.02; // seconds
const double KEEP_LANE_MULTIPLIER = 2;
const double SLOW_DOWN_MULTIPLIER = 6; //10
const double CHANGE_LANE_MULTIPLIER = 12;
const double START_MULTIPLIER = 15;
const double TRAJECTORY_POINTS = 15;
const double COLLISION_DISTANCE = 30.0; // meters
const double BUFFER_DISTANCE = 35.0; // meters
const double LANE_SPEED_COST_WEIGHT = 3.0;
const double CARS_PER_LANE_COST_WEIGHT = 1.0;
const double LANE_DISTANCE_COST_WEIGHT = 20.0;

#endif
