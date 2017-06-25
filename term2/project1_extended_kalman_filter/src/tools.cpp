#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {

  VectorXd rmse(4);
	rmse << 0, 0, 0, 0;

  if (estimations.size() == 0) {
    std::cout << "CalculateRMSE - Error: Invalid estimation size" << std::endl;
    return rmse;
  }

  if (ground_truth.size() == 0) {
    std::cout << "CalculateRMSE - Error: Invalid groud truth size" << std::endl;
    return rmse;
  }

  if (estimations.size() != ground_truth.size()) {
    std::cout << "CalculateRMSE - Error: Size mismatch" << std::endl;
    return rmse;
  }

  for (int i = 0; i < estimations.size(); ++i) {
        VectorXd error = estimations[i] - ground_truth[i];
        error = error.array() * error.array();
        rmse += error;
	}

	rmse = rmse / estimations.size();
	rmse = rmse.array().sqrt();

	return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd Hj = MatrixXd(3, 4);
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  float c1 = px * px + py * py;
  if (fabs(c1) < 0.0001) {
      std::cout << "CalculateJacobian - Error: c1: " << c1 << endl;
      return Hj;
  }

  float c2 = sqrt(c1);
  float c3 = c1 * c2;

  Hj(0, 0) = px / c2;
  Hj(0, 1) = py / c2;
  Hj(0, 2) = 0;
  Hj(0, 3) = 0;
  Hj(1, 0) = -py / c1;
  Hj(1, 1) = px / c1;
  Hj(1, 2) = 0;
  Hj(1, 3) = 0;
  Hj(2, 0) = py * (vx * py - vy * px) / c3;
  Hj(2, 1) = px * (vy * px - vx * py) / c3;
  Hj(2, 2) = px / c2;
  Hj(2, 3) = py / c2;

  return Hj;
}
