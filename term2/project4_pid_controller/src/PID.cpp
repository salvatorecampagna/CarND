#include "PID.h"

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
  /* Initialize PID gains */
  this->Kp = Kp;
  this->Ki = Ki;
  this->Kd = Kd;

  /* Initialize PID errors */
  this->p_error = 0;
  this->i_error = 0;
  this->d_error = 0;
}

void PID::UpdateError(double cte) {
  /* Compute PID errors at each iteration */
  
  /* Derivative error */
  d_error = cte - p_error;
  /* Proportional error */
  p_error = cte;
  /* Integral error */
  i_error += cte;
}

double PID::TotalError() {
  /* Compute PID total error as sum of proportional, integral and derivative components */
  return -Kp * p_error - Ki * i_error - Kd * d_error;
}
