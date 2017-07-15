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