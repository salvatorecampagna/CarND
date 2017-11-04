#include "planner.h"

const double MAX_D = 12.0; // meters
const double MAX_S = 6914.14925765991; // meters
const double SPEED_LIMIT = 19; // meters / second
const double DELTA_T = 0.02; // seconds
const double KEEP_LANE_MULTIPLIER = 2;
const double SLOW_DOWN_MULTIPLIER = 3;
const double CHANGE_LANE_MULTIPLIER = 4;
const double START_MULTIPLIER = 4;
const double TRAJECTORY_POINTS = 50;

// Convert a state to string
std::string state_to_string(STATE state)
{
  std::string state_string;

  switch (state)
  {
    case STATE::KEEP_LANE_STATE:
      state_string = "keep lane";
      break;

    case STATE::CHANGE_LEFT_STATE:
      state_string = "change lane left";
      break;

    case STATE::CHANGE_RIGHT_STATE:
      state_string = "change lane right";
      break;

    default:
      state_string = "unknown";
  }

  return state_string;
}

// Convert a lane to string
std::string lane_to_string(LANE lane)
{
  std::string lane_string;

  switch(lane)
  {
    case LANE::LEFT_LANE:
      lane_string = "left lane";
      break;

    case LANE::RIGHT_LANE:
      lane_string = "right lane";
      break;

    case LANE::CENTER_LANE:
      lane_string = "center lane";
      break;

    default:
      lane_string = "unknown";
  }

  return lane_string;
}

// Get the lane given the distance from center of the road
LANE get_lane(double d)
{
  LANE lane;

  // Left lane 0 - 4 meters
  if (d < 4.0)
  {
    lane = LANE::LEFT_LANE;
  }
  // Center lane 4 - 8 meters
  else if ((d >= 4.0) && (d < 8.0))
  {
    lane = LANE::CENTER_LANE;
  }
  // Right lane 8 - 12 meters
  else
  {
    lane = LANE::RIGHT_LANE;
  }

  return lane;
}

// Get the lane center given the lane
double get_lane_center_d(LANE lane)
{
  double d;

  switch (lane)
  {
    case LANE::LEFT_LANE:
      d = 2.0; // Left lane center
      break;

    case LANE::CENTER_LANE:
      d = 6.0; // Center lane center
      break;

    case LANE::RIGHT_LANE:
      d = 10.0; // Right lane ceter
      break;

    default:
      d = 6.0; // Center lane center
  }

  return d;
}

// Get the lane ceter given the distance
double get_lane_center_d(double distance){
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

Planner::Planner()
{
  this->state = STATE::START_STATE;
}

// Jerk Minimizing Trajectory
std::vector<double> Planner::JMT(std::vector<double> start, std::vector<double> end, double t) {
  std::vector<double> coeffs;
  double t2 = t * t;
  double t3 = t2 * t;
  double t4 = t3 * t;
  double t5 = t4 * t;
  /*
  Calculate the Jerk Minimizing Trajectory.
  The Jerk Minimizing Trajectory is described by a polynomila of degree 5
  (which has 6 coefficients).

  INPUTS
  ------
  start - the vehicles start location given as a three elements array
  [s, s_dot, s_dot_dot]

  end - the desired end state for the vehicle given as a three elements array
  [d, d_dot, d_dot_dot]

  t - The time, in seconds, taken by the menuver.

  OUTPUT
  ------
  a 6 elemnts array where each value corresponds to a coefficent in the polynomial
  s(t) = a0 + a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5
  */

  // Prepare matrix A with coefficents
  Eigen::MatrixXd A(3, 3);
  Eigen::MatrixXd b(3, 1);

  A << t3, t4, t5,
       3 * t2, 4 * t3, 5 * t4,
       6 * t, 12 * t2, 20 * t3;

  b << end[0] - (start[0] + start[1] * t + 0.5 * start[2] * t2),
       end[1] - (start[1] + start[2] * t),
       end[2] - start[2];

  Eigen::MatrixXd Ainv = A.inverse();
  Eigen::MatrixXd c = Ainv * b;

  coeffs.push_back(start[0]);
  coeffs.push_back(start[1]);
  coeffs.push_back(0.5 * start[2]);
  coeffs.push_back(c.data()[0]);
  coeffs.push_back(c.data()[1]);
  coeffs.push_back(c.data()[2]);

  return coeffs;
}

// Computes a trajectory in Frenet coordinates given the action decided by the FSM
void Planner::compute_trajectory(Map& map, std::vector<std::vector<double>>& trajectory)
{
  std::vector <double> xy;
  double t, s_next, d_next, s_mod, d_mod;
  double T = this->n * DELTA_T;

  // Compute Jerk Minimizing Trajectory s(t) and d(t)
  std::vector<double> s_poly = this->JMT(this->start_s, this->end_s, T);
  std::vector<double> d_poly = this->JMT(this->start_d, this->end_d, T);

  for(int i = 0; i < n; i++)
  {
    t = DELTA_T * i;
    s_next = 0.0;
    d_next = 0.0;

    // Evaluate s(t) at time t = DELTA_T * i
    for (int i = 0; i < s_poly.size(); i++)
    {
      s_next += s_poly[i] * pow(t, i);
    }

    // Evaluate d(t) at time t = DELTA_T * i
    for (int i = 0; i < d_poly.size(); i++)
    {
      d_next += d_poly[i] * pow(t, i);
    }

    // Wrap around s and d coordinates (the track is cyclic)
    s_mod = fmod(s_next, MAX_S);
    d_mod = fmod(d_next, MAX_D);

    // Convert Frenet coordinates to XY coordinates
    xy = map.getXY(s_mod, d_mod);

    trajectory[0].push_back(xy[0]);
    trajectory[1].push_back(xy[1]);
  }

}

// FSM: given the map, the vehicle status and road status
// determine the next action which could be: keep lane, slow down or
// change lane (left or right)
void Planner::new_trajectory(Map& map, Road& road, Vehicle& car, std::vector<std::vector<double>>& trajectory)
{
  int trajectory_size = trajectory[0].size();
  this->recompute_trajectory = false;

  if (trajectory_size < TRAJECTORY_POINTS)
  {
    this->recompute_trajectory = true;

    // Start state trajectory
    if (this->state == STATE::START_STATE)
    {
      this->start(car);
    }
    // KEEP_LANE_STATE
    else if (this->state == STATE::KEEP_LANE_STATE)
    {
      if (road.is_lane_safe(car, car.lane()))
      {
        // No car within BUFFER_DISTANCE, keep lane
        this->keep_lane(car);
      }
      else
      {
        // There is a car within BUFFER_DISTANCE.
        // Either slow down or change lane
        LANE target_lane = road.find_target_lane(car);
        if (target_lane == car.lane())
        {
          // Target lane is the current lane, slow down
          this->slow_down(car);
        }
        else
        {
          // Target lane is a different lane, change lane
          this->change_lane(car, target_lane);
        }
      }
    }
    // CHANGE_LANE_LEFT or CHANGE_LANE_RIGHT
    else
    {
      LANE target_lane = get_lane(car.get_previous_d()[0]);
      if(road.is_lane_safe(car, target_lane))
      {
        // No car within BUFFER_DISTANCE, keep lane
        this->keep_lane(car);
      }
      else
      {
        // Car within BUFFER_DISTANCE, slow down
        this->slow_down(car);
      }
    }
  }

  if (this->recompute_trajectory)
  {
    // Recompute trajectory
    this->compute_trajectory(map, trajectory);
  }
}

// Update status given the current state and the target lane
void Planner::set_state(LANE curr_lane, LANE target_lane)
{
  if (curr_lane == target_lane)
  {
    this->state = STATE::KEEP_LANE_STATE;
  }
  else
  {
    if(curr_lane == LANE::LEFT_LANE)
    {
      this->state = STATE::CHANGE_RIGHT_STATE;
    }
    else if(curr_lane == LANE::RIGHT_LANE)
    {
      this->state = STATE::CHANGE_LEFT_STATE;
    }
    else
    {
      if(target_lane == LANE::LEFT_LANE)
      {
        this->state = STATE::CHANGE_LEFT_STATE;
      }
      else
      {
        this->state = STATE::CHANGE_RIGHT_STATE;
      }
    }
  }
}

// Start state action
void Planner::start(Vehicle& car)
{
  std::cout << "[action] start" << std::endl;
  std::cout << "(s: " << car.get_s() << " d: " << car.get_d() << " v: " << car.get_v() << ")" << std::endl;

  this->n = START_MULTIPLIER * TRAJECTORY_POINTS;
  double target_v = SPEED_LIMIT / 2.2;
  double target_s = car.get_s() + n * DELTA_T * target_v;

  this->start_s = {car.get_s(), car.get_v(), 0.0};
  this->end_s= {target_s, target_v, 0.0};
  this->start_d = {get_lane_center_d(car.lane()), 0.0, 0.0};
  this->end_d = {get_lane_center_d(car.lane()), 0.0, 0.0};

  car.set_previous_s(this->end_s);
  car.set_previous_d(this->end_d);

  LANE curr_lane = car.lane();
  LANE target_lane = car.lane();
  this->set_state(curr_lane, target_lane);
}

// Keep lane action
void Planner::keep_lane(Vehicle& car)
{
  std::cout << "[action] keep lane: " << lane_to_string(car.lane()) << std::endl;
  std::cout << "(s: " << car.get_s() << " d: " << car.get_d() << " v: " << car.get_v() << ")" << std::endl;

  this->n = KEEP_LANE_MULTIPLIER * TRAJECTORY_POINTS;
  double target_v = std::min(car.get_previous_s()[1] * 1.20, SPEED_LIMIT);
  double target_s = car.get_previous_s()[0] + n * DELTA_T * target_v;
  double target_d = get_lane_center_d(car.get_previous_d()[0]);

  this->start_s = {car.get_previous_s()[0], car.get_previous_s()[1], car.get_previous_s()[2]};
  this->end_s = {target_s, target_v, 0.0};
  this->start_d = {get_lane_center_d(car.get_previous_d()[0]), 0.0, 0.0};
  this->end_d = {target_d, 0.0, 0.0};

  car.set_previous_s(this->end_s);
  car.set_previous_d(this->end_d);

  LANE curr_lane = get_lane(car.get_previous_d()[0]);
  LANE target_lane = get_lane(car.get_previous_d()[0]);
  this->set_state(curr_lane, target_lane);
}

// Slow down action
void Planner::slow_down(Vehicle& car){
  std::cout << "[action] slow down: " << car.get_v() << std::endl;
  std::cout << "(s: " << car.get_s() << " d: " << car.get_d() << " v: " << car.get_v() << ")" << std::endl;

  this->n = SLOW_DOWN_MULTIPLIER * TRAJECTORY_POINTS;
  this->recompute_trajectory = true;
  double target_v = std::max(car.get_previous_s()[1] * 0.90, SPEED_LIMIT / 2.0);
  double target_s = car.get_previous_s()[0] + n * DELTA_T * target_v;
  double target_d = get_lane_center_d(car.get_previous_d()[0]);

  this->start_s = {car.get_previous_s()[0], car.get_previous_s()[1], car.get_previous_s()[2]};
  this->end_s = {target_s, target_v, 0.0};
  this->start_d = {get_lane_center_d(car.get_previous_d()[0]), 0.0, 0.0};
  this->end_d = {target_d, 0.0, 0.0};

  car.set_previous_s(this->end_s);
  car.set_previous_d(this->end_d);

  LANE curr_lane = get_lane(car.get_previous_d()[0]);
  LANE target_lane = get_lane(car.get_previous_d()[0]);
  this->set_state(curr_lane, target_lane);
}

// Change lane action
void Planner::change_lane(Vehicle& car, LANE new_lane){
  std::cout << "[action] change lane: " << lane_to_string(car.lane());
  std::cout << " => " << lane_to_string(new_lane) << std::endl;
  std::cout << "(s: " << car.get_s() << " d: " << car.get_d() << " v: " << car.get_v() << ")" << std::endl;

  this->n = CHANGE_LANE_MULTIPLIER * TRAJECTORY_POINTS;
  this->recompute_trajectory = true;
  double target_v = car.get_previous_s()[1];
  double target_s = car.get_previous_s()[0] + n * DELTA_T * target_v;
  double target_d = get_lane_center_d(new_lane);

  this->start_s = {car.get_previous_s()[0], car.get_previous_s()[1], car.get_previous_s()[2]};
  this->end_s = {target_s, target_v, 0.0};
  this->start_d = {get_lane_center_d(car.get_previous_d()[0]), 0.0, 0.0};
  this->end_d = {target_d, 0.0, 0.0};

  car.set_previous_s(this->end_s);
  car.set_previous_d(this->end_d);

  LANE curr_lane = get_lane(car.get_previous_d()[0]);
  LANE target_lane = get_lane(target_d);
  this->set_state(curr_lane, target_lane);
}
