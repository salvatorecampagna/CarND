#include "kalman_filter.h"
#include <iostream>

#define M_PI 3.14159265358979323846

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  MatrixXd Ft = F_.transpose();
  x_ = F_ * x_;
  P_ = F_ * P_ * Ft + Q_;

  //std::cout << "x_ = " << x_ << std::endl;
  //std::cout << "P_ = " << P_ << std::endl;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd y = z - H_ * x_;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  x_ = x_ + (K * y);
  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
  P_ = (I - K * H_) * P_;

  //std::cout << "x_ = " << x_ << std::endl;
  //std::cout << "P_ = " << P_ << std::endl;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);

  if (fabs(px) < 0.0001) {
    std::cout << "UpdateEKF () - Error: px: " << px << std::endl;
    px = 0.0001;
  }

  if (fabs(py) < 0.0001) {
    std::cout << "UpdateEKF () - Error: py: " << py << std::endl;
    py = 0.0001;
  }

  if (fabs(vx) < 0.0001) {
    std::cout << "UpdateEKF () - Error: vx: " << vx << std::endl;
    vx = 0.0001;
  }

  if (fabs(vy) < 0.0001) {
    std::cout << "UpdateEKF () - Error: vy: " << vy << std::endl;
    vy = 0.0001;
  }

  float rho = sqrt(px * px + py * py);
  if (fabs(rho) < 0.0001) {
    std::cout << "UpdateEKF () - Error: rho: " << rho << std::endl;
    rho = 0.0001;
  }

  float phi = atan2(py, px);
  if (phi < -2 * M_PI || phi > 2 * M_PI) {
    std::cout << "UpdateEKF () - Error: phi: " << phi << std::endl;
  }

  float rho_dot = (px * vx + py * vy) / rho;
  if (fabs(rho_dot) < 0.0001) {
    std::cout << "UpdateEKF () - Error: rho_dot: " << rho_dot << std::endl;
    rho_dot = 0.0001;
  }

  VectorXd hx(3);
  hx << rho, phi, rho_dot;

  VectorXd y = z - hx;
  //adjust phi in range (-pi, pi)
  while (y(1) < -M_PI)
    y(1) += 2 * M_PI;
  while (y(1) > M_PI)
    y(1) -= 2 * M_PI;

  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  x_ = x_ + (K * y);
  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
  P_ = (I - K * H_) * P_;

  //std::cout << "x_ = " << x_ << std::endl;
  //std::cout << "P_ = " << P_ << std::endl;
}
