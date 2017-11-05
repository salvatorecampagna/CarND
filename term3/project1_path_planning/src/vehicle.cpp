#include "vehicle.h"

Vehicle::Vehicle()
{
  this->id = -1;
}

Vehicle::Vehicle(int id, double x, double y, double v, double s, double d)
{
  this->id = id;
  this->x = x;
  this->y = y;
  this->v = v;
  this->s = s;
  this->d = d;
}

int Vehicle::get_id()
{
  return this->id;
}

double Vehicle::get_x()
{
  return this->x;
}

double Vehicle::get_y()
{
  return this->y;
}

double Vehicle::get_v()
{
  return this->v;
}

double Vehicle::get_s()
{
  return this->s;
}

double Vehicle::get_d()
{
  return this->d;
}

unsigned int Vehicle::lane()
{
  unsigned int lane;

  if (this->d < 4.0)
  {
    lane = Lane::LEFT_LANE;
  }
  else if ((this->d >= 4.0) && (this->d < 8.0))
  {
    lane = Lane::CENTER_LANE;
  }
  else
  {
    lane = Lane::RIGHT_LANE;
  }

  return lane;
}

void Vehicle::update_vehicle_status(double x, double y, double v, double s, double d, double yaw)
{
  this->x = x;
  this->y = y;
  this->v = v;
  this->s = s;
  this->d = d;
  this->yaw = yaw;
}

void Vehicle::set_previous_s(std::vector<double> previous_s)
{
  this->previous_s = previous_s;
}

void Vehicle::set_previous_d(std::vector<double> previous_d)
{
  this->previous_d = previous_d;
}

std::vector<double> Vehicle::get_previous_s()
{
  return this->previous_s;
}

std::vector<double> Vehicle::get_previous_d()
{
  return this->previous_d;
}
