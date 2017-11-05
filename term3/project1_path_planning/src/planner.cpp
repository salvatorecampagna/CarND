#include "planner.h"
#include "params.h"
#include "lane.h"

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

Planner::Planner()
{
  this->state = STATE::START_STATE;
}

// Jerk Minimizing Trajectory
std::vector<double> Planner::jerk_min_trajectory(std::vector<double> start, std::vector<double> end, double t) {
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
  std::vector<double> s_poly = this->jerk_min_trajectory(this->start_s, this->end_s, T);
  std::vector<double> d_poly = this->jerk_min_trajectory(this->start_d, this->end_d, T);


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
      this->start(road, car);
    }
    // KEEP_LANE_STATE
    else if (this->state == STATE::KEEP_LANE_STATE)
    {
      if (road.is_lane_safe(car, car.lane()))
      {
        // No car within BUFFER_DISTANCE, keep lane
        this->keep_lane(road, car);
      }
      else
      {
        // There is a car within BUFFER_DISTANCE.
        // Either slow down or change lane
        unsigned int target_lane = road.find_target_lane(car);
        if (target_lane == car.lane())
        {
          // Target lane is the current lane, slow down
          this->slow_down(road, car);
        }
        else
        {
          // Target lane is a different lane, change lane
          this->change_lane(road, car, target_lane);
        }
      }
    }
    // CHANGE_LANE_LEFT or CHANGE_LANE_RIGHT
    else
    {
      unsigned int target_lane = Lane::get_lane(car.get_previous_d()[0]);
      if(road.is_lane_safe(car, target_lane))
      {
        // No car within BUFFER_DISTANCE, keep lane
        this->keep_lane(road, car);
      }
      else
      {
        // Car within BUFFER_DISTANCE, slow down
        this->slow_down(road, car);
      }
    }
  }

  if (this->recompute_trajectory)
  {
    // Recompute trajectory
    this->compute_trajectory(map, trajectory);
  }
}

// Adjust the speed in keep lane state
double Planner::acceleration(Vehicle& car, double distance, double speed)
{
  double car_v = car.get_v();
  double a = 1.04;

  if (car_v > 45)
  {
    a = 1.02;
  }
  if (car_v > 47)
  {
    a = 1.01;
  }
  if (car_v > 48)
  {
    a = 0.98;
  }
  if (car_v > 49)
  {
    a = 0.95;
  }

  if (distance < COLLISION_DISTANCE)
  {
    a = 0.95;
  }

  if (distance < BUFFER_DISTANCE)
  {
    a = 1.0;
  }

  if (distance < 1.10 * BUFFER_DISTANCE)
  {
    a = 1.01;
  }

  return std::min(car.get_previous_s()[1] * a, std::max(speed, SPEED_LIMIT));
}

// Adjust the speed in while slowing down
double Planner::brake(Vehicle& car, double distance, double speed)
{
  double v = 0.98 * speed;

  if (distance < 0.5 * COLLISION_DISTANCE)
  {
    v = speed * 0.85;
  }
  else if (distance < 0.80 * COLLISION_DISTANCE)
  {
    v = speed * 0.90;
  }
  else if (distance < COLLISION_DISTANCE)
  {
    v = speed * 0.93;
  }
  else if (distance < BUFFER_DISTANCE)
  {
    v = speed * 0.95;
  }

  return std::max(v, car.get_previous_s()[1] * 0.80);
}

double Planner::overtake(Vehicle& car, double distance, double curr_lane_speed, double target_lane_speed)
{
  double v = car.get_previous_s()[1];

  if (distance < BUFFER_DISTANCE)
  {
    v = std::min(0.50 * (curr_lane_speed + target_lane_speed), SPEED_LIMIT);
  }

  return v;
}

// Start state action
void Planner::start(Road& road, Vehicle& car)
{
  double s = car.get_s();
  double d = car.get_d();
  double v = car.get_v();
  std::cout << "-------------------------------------------------" << std::endl;
  std::cout << "[action] start" << std::endl;
  std::cout << "(s: " << s << " d: " << d << " v: " << v << ")" << std::endl;

  this->n = START_MULTIPLIER * TRAJECTORY_POINTS;
  double front_v = road.get_front_vehicle_speed(car);
  double target_v = std::min(front_v, SPEED_LIMIT / 2.2);
  double target_s = car.get_s() + n * DELTA_T * target_v;

  this->start_s = {car.get_s(), car.get_v(), 0.0};
  this->end_s= {target_s, target_v, 0.0};
  this->start_d = {Lane::get_lane_center_d(car.lane()), 0.0, 0.0};
  this->end_d = {Lane::get_lane_center_d(car.lane()), 0.0, 0.0};

  car.set_previous_s(this->end_s);
  car.set_previous_d(this->end_d);

  unsigned int curr_lane = car.lane();
  unsigned int target_lane = car.lane();
  this->update_state(curr_lane, target_lane);
}

// Keep lane action
void Planner::keep_lane(Road& road, Vehicle& car)
{
  double s = car.get_s();
  double d = car.get_d();
  double v = car.get_v();
  std::cout << "-------------------------------------------------" << std::endl;
  std::cout << "[action] keep lane: " << Lane::lane_to_string(car.lane()) << std::endl;
  std::cout << "(s: " << s << " d: " << d << " v: " << v << ")" << std::endl;

  this->n = KEEP_LANE_MULTIPLIER * TRAJECTORY_POINTS;
  double front_v = road.get_front_vehicle_speed(car);
  double front_distance = road.get_front_vehicle_distance(car);
  double target_v = acceleration(car, front_distance, front_v);
  double target_s = car.get_previous_s()[0] + n * DELTA_T * target_v;
  double target_d = Lane::get_lane_center_d(car.get_previous_d()[0]);

  this->start_s = {car.get_previous_s()[0], car.get_previous_s()[1], car.get_previous_s()[2]};
  this->end_s = {target_s, target_v, 0.0};
  this->start_d = {Lane::get_lane_center_d(car.get_previous_d()[0]), 0.0, 0.0};
  this->end_d = {target_d, 0.0, 0.0};

  car.set_previous_s(this->end_s);
  car.set_previous_d(this->end_d);

  unsigned int curr_lane = Lane::get_lane(car.get_previous_d()[0]);
  unsigned int target_lane = Lane::get_lane(car.get_previous_d()[0]);
  this->update_state(curr_lane, target_lane);
}

// Slow down action
void Planner::slow_down(Road& road, Vehicle& car)
{
  double s = car.get_s();
  double d = car.get_d();
  double v = car.get_v();
  std::cout << "-------------------------------------------------" << std::endl;
  std::cout << "[action] slow down: " << car.get_v() << std::endl;
  std::cout << "(s: " << s << " d: " << d << " v: " << v << ")" << std::endl;

  this->n = SLOW_DOWN_MULTIPLIER * TRAJECTORY_POINTS;
  this->recompute_trajectory = true;
  double front_v = road.get_front_vehicle_speed(car);
  double front_distance = road.get_front_vehicle_distance(car);
  double target_v =  brake(car, front_distance, front_v);
  double target_s = car.get_previous_s()[0] + n * DELTA_T * target_v;
  double target_d = Lane::get_lane_center_d(car.get_previous_d()[0]);

  this->start_s = {car.get_previous_s()[0], car.get_previous_s()[1], car.get_previous_s()[2]};
  this->end_s = {target_s, target_v, 0.0};
  this->start_d = {Lane::get_lane_center_d(car.get_previous_d()[0]), 0.0, 0.0};
  this->end_d = {target_d, 0.0, 0.0};

  car.set_previous_s(this->end_s);
  car.set_previous_d(this->end_d);

  unsigned int curr_lane = Lane::get_lane(car.get_previous_d()[0]);
  unsigned int target_lane = Lane::get_lane(car.get_previous_d()[0]);
  this->update_state(curr_lane, target_lane);
}

// Change lane action
void Planner::change_lane(Road& road, Vehicle& car, unsigned int new_lane)
{
  double s = car.get_s();
  double d = car.get_d();
  double v = car.get_v();
  std::cout << "-------------------------------------------------" << std::endl;
  std::cout << "[action] change lane: " << Lane::lane_to_string(car.lane());
  std::cout << " => " << Lane::lane_to_string(new_lane) << std::endl;
  std::cout << "(s: " << s << " d: " << d << " v: " << v << ")" << std::endl;

  this->n = CHANGE_LANE_MULTIPLIER * TRAJECTORY_POINTS;
  this->recompute_trajectory = true;
  double front_v = road.get_front_vehicle_speed(car);
  double front_distance = road.get_front_vehicle_distance(car);
  double next_lane_v = road.get_front_vehicle_speed(car, new_lane);
  double target_v = overtake(car, front_distance, front_v, next_lane_v);
  double target_s = car.get_previous_s()[0] + n * DELTA_T * target_v;
  double target_d = Lane::get_lane_center_d(new_lane);

  this->start_s = {car.get_previous_s()[0], car.get_previous_s()[1], car.get_previous_s()[2]};
  this->end_s = {target_s, target_v, 0.0};
  this->start_d = {Lane::get_lane_center_d(car.get_previous_d()[0]), 0.0, 0.0};
  this->end_d = {target_d, 0.0, 0.0};

  car.set_previous_s(this->end_s);
  car.set_previous_d(this->end_d);

  unsigned int curr_lane = Lane::get_lane(car.get_previous_d()[0]);
  unsigned int target_lane = Lane::get_lane(target_d);
  this->update_state(curr_lane, target_lane);
}

// Update status given the current state and the target lane
void Planner::update_state(unsigned int curr_lane, unsigned int target_lane)
{
  if (curr_lane == target_lane)
  {
    this->state = STATE::KEEP_LANE_STATE;
  }
  else
  {
    if(curr_lane == Lane::LEFT_LANE)
    {
      this->state = STATE::CHANGE_RIGHT_STATE;
    }
    else if(curr_lane == Lane::RIGHT_LANE)
    {
      this->state = STATE::CHANGE_LEFT_STATE;
    }
    else
    {
      if(target_lane == Lane::LEFT_LANE)
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
