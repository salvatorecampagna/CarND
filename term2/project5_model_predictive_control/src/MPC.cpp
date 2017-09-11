#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"

using CppAD::AD;

// T = N * dt is the prediction horizon
// N: number of timesteps in the horizon
// dt: time between control command actuations
size_t N = 10;
double dt = 0.1;

// Reference speed
double ref_v = 100.0;
// Reference Cross-Track Error
double ref_cte = 0.0;
// Reference Heading Error
double ref_epsi = 0.0;

size_t x_start = 0;
size_t y_start = x_start + N;
size_t psi_start = y_start + N;
size_t v_start = psi_start + N;
size_t cte_start = v_start + N;
size_t epsi_start = cte_start + N;
size_t delta_start = epsi_start + N;
size_t a_start = delta_start + N - 1;

// This value is obtained by measuring the radius formed by running the vehicle in the
// simulator around in a circle with a constant steering angle and velocity on a
// flat terrain.
//
// Lf was tuned until the the radius formed by simulating the model
// presented matched the previous radius.
//
// This is the length from front to Center of Gfravity that has a similar radius.
const double Lf = 2.67;

class FG_eval {
 public:
  // Fitted polynomial coefficients
  Eigen::VectorXd coeffs;
  FG_eval(Eigen::VectorXd coeffs) { this->coeffs = coeffs; }

  typedef CPPAD_TESTVECTOR(AD<double>) ADvector;
  void operator()(ADvector& fg, const ADvector& vars) {
    // `fg` a vector of the cost constraints, `vars` is a vector of variable values (state & actuators)
    
    // The idea is to find the value for N steering control inputs and
    // N acceleration control inputs minimizing the cost function
    // (even though, at the end, only the first steering control
    // input and acceleration control input will be used)

    // fg[0] stores the cost
    fg[0] = 0;

    // Add CTE, heading and speed to the cost
    for (size_t t = 0; t < N; t++) {
      // Add Cross-Track Error (CTE) to the cost
      fg[0] += 2500 * CppAD::pow(vars[cte_start + t] - ref_cte, 2);
      // Add Heading Error to the cost
      fg[0] += 2500 * CppAD::pow(vars[epsi_start + t] - ref_epsi, 2);
      // Add reference speed to cost to avoid the car stopping
      fg[0] += 0.5 * CppAD::pow(vars[v_start + t] - ref_v, 2);
    }

    // Add control input to the cost
    for (size_t t = 0; t < N - 1; t++) {
      // Add steering control to cost
      fg[0] += 5 * CppAD::pow(vars[delta_start + t], 2);
      // Add acceleration control to the cost
      fg[0] += 5 * CppAD::pow(vars[a_start + t], 2);
    }

    // Add control input difference to the cost to have a smoother behavior
    for (size_t t = 0; t < N - 2; t++) {
      // Add steering difference to the cost (steering(t+1) - steering(t))
      fg[0] += 200 * CppAD::pow(vars[delta_start + t + 1] - vars[delta_start + t], 2);
      // Add acceleration difference to the cost (acceleration(t+1) - acceleration(t))
      fg[0] += 10 * CppAD::pow(vars[a_start + t + 1] - vars[a_start + t], 2);
    }

    // Initial constraints
    fg[x_start + 1] = vars[x_start];
    fg[y_start + 1] = vars[y_start];
    fg[psi_start + 1] = vars[psi_start];
    fg[v_start + 1] = vars[v_start];
    fg[cte_start + 1] = vars[cte_start];
    fg[epsi_start + 1] = vars[epsi_start];

    // The rest of the constraints
    for (size_t t = 0; t < N - 1; t++)
    {
      // The state at time t+1
      AD<double> x1 = vars[x_start + t + 1];
      AD<double> y1 = vars[y_start + t + 1];
      AD<double> psi1 = vars[psi_start + t + 1];
      AD<double> v1 = vars[v_start + t + 1];
      AD<double> cte1 = vars[cte_start + t + 1];
      AD<double> epsi1 = vars[epsi_start + t + 1];

      // The state at time t
      AD<double> x0 = vars[x_start + t];
      AD<double> y0 = vars[y_start + t];
      AD<double> psi0 = vars[psi_start + t];
      AD<double> v0 = vars[v_start + t];
      AD<double> cte0 = vars[cte_start + t];
      AD<double> epsi0 = vars[epsi_start + t];

      // Only consider the actuation at time t.
      AD<double> delta0 = vars[delta_start + t];
      AD<double> a0 = vars[a_start + t];

      AD<double> f0 = coeffs[0] + coeffs[1] * x0 + coeffs[2] * x0 * x0 + coeffs[3] * x0 * x0 *x0;
      AD<double> psides0 = CppAD::atan(3 * coeffs[3] * x0 * x0 + 2 * coeffs[2] * x0 + coeffs[1]);

      // Next state
      fg[x_start + t + 2] = x1 - (x0 + v0 * CppAD::cos(psi0) * dt);
      fg[y_start + t + 2] = y1 - (y0 + v0 * CppAD::sin(psi0) * dt);
      fg[psi_start + t + 2] = psi1 - (psi0 + v0 * delta0 / Lf * dt);
      fg[v_start + t + 2] = v1 - (v0 + a0 * dt);
      fg[cte_start + t + 2] = cte1 - ((f0 - y0) + (v0 * CppAD::sin(epsi0) * dt));
      fg[epsi_start + t + 2] = epsi1 - ((psi0 - psides0) + v0 * delta0 / Lf * dt);
    }
  }
};

//
// MPC class definition implementation.
//
MPC::MPC() {}
MPC::~MPC() {}

vector<double> MPC::Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs) {
  bool ok = true;
  size_t i;
  typedef CPPAD_TESTVECTOR(double) Dvector;

  // State variables
  double x = state[0];
  double y = state[1];
  double psi = state[2];
  double v = state[3];
  double cte = state[4];
  double epsi = state[5];

  // TNumber of model variables (includes both states and inputs).
  // For example: If the state is a 4 element vector, the actuators is a 2
  // element vector and there are 10 timesteps. The number of variables is:
  //
  // 4 * 10 + 2 * 9

  // We have 6 state variables (x, y, psi, v, cte and epsi)
  // And we have 2 control inputs, steering angle (delta) and acceleration (a) 
  size_t n_vars = 6 * N + 2 * (N - 1);
  // Number of constraints
  size_t n_constraints = 6 * N;

  // Initial value of the independent variables
  // SHOULD BE 0 besides initial state
  Dvector vars(n_vars);
  for (i = 0; i < n_vars; i++) {
    vars[i] = 0.0;
  }

  Dvector vars_lowerbound(n_vars);
  Dvector vars_upperbound(n_vars);
  // Set lower and upper limits for variables

  // Initial variable values
  vars[x_start] = x;
  vars[y_start] = y;
  vars[psi_start] = psi;
  vars[v_start] = v;
  vars[cte_start] = cte;
  vars[epsi_start] = epsi;

  // Lower and upper bound for all variables other than
  // steering angle and acceleration
  for (i = 0; i < delta_start; i++)
  {
    vars_lowerbound[i] = -1.0e19; // Unlimited
    vars_upperbound[i] = 1.0e19; // Unlimited
  }

  // Lower and upper bound for the steering
  // angle control input
  for (i = delta_start; i < a_start; i++)
  {
    vars_lowerbound[i] = -0.436332313 * Lf; // -25 degrees
    vars_upperbound[i] = 0.436332313 * Lf;  // 25 degrees
  }

  // Lower and upper bound for the acceleration
  // control input
  for (i = a_start; i < n_vars; i++)
  {
    vars_lowerbound[i] = -1.0; // -1.0 full brake
    vars_upperbound[i] = 1.0; // 1.0 full acceleration
  }

  // Lower and upper limits for the constraints
  // Should be 0 besides initial state.
  Dvector constraints_lowerbound(n_constraints);
  Dvector constraints_upperbound(n_constraints);
  for (i = 0; i < n_constraints; i++) {
    constraints_lowerbound[i] = 0;
    constraints_upperbound[i] = 0;
  }

  // Setting constrainrs
  // stete(t+1) = state(t) => state(t+1) - state(t) = 0
  constraints_lowerbound[x_start] = x;
  constraints_lowerbound[y_start] = y;
  constraints_lowerbound[psi_start] = psi;
  constraints_lowerbound[v_start] = v;
  constraints_lowerbound[cte_start] = cte;
  constraints_lowerbound[epsi_start] = epsi;

  constraints_upperbound[x_start] = x;
  constraints_upperbound[y_start] = y;
  constraints_upperbound[psi_start] = psi;
  constraints_upperbound[v_start] = v;
  constraints_upperbound[cte_start] = cte;
  constraints_upperbound[epsi_start] = epsi;

  // object that computes objective and constraints
  FG_eval fg_eval(coeffs);

  // Options for IPOPT solver
  std::string options;
  // Uncomment this if you'd like more print information
  options += "Integer print_level  0\n";
  // NOTE: Setting sparse to true allows the solver to take advantage
  // of sparse routines, this makes the computation MUCH FASTER. If you
  // can uncomment 1 of these and see if it makes a difference or not but
  // if you uncomment both the computation time should go up in orders of
  // magnitude.
  options += "Sparse  true        forward\n";
  options += "Sparse  true        reverse\n";
  // NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
  // Change this as you see fit.
  options += "Numeric max_cpu_time          0.5\n";

  CppAD::ipopt::solve_result<Dvector> solution;

  CppAD::ipopt::solve<Dvector, FG_eval>(
      options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
      constraints_upperbound, fg_eval, solution);

  // Check some of the solution values
  ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

  // Cost
  auto cost = solution.obj_value;
  std::cout << "Cost: " << cost << std::endl;

  std::vector<double> result;

  result.push_back(solution.x[delta_start]);
  result.push_back(solution.x[a_start]);

  // Return the actuator values for steering angle and acceleration
  // Only the first control input values are returned (one for the steering
  // angle and the other for the acceleration, other values are discarded)
  return result;
}
