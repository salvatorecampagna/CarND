#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);
  x_ << 0, 0, 0, 0, 0;

  // initial covariance matrix
  P_ = MatrixXd::Identity(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  //std_a_ = 30; //TODO: tune this value
  std_a_ = 3.0;

  // Process noise standard deviation yaw acceleration in rad/s^2
  //std_yawdd_ = 30; //TODO: tune this value
  std_yawdd_ = 2.0;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // Number of state variables (size of x_)
  n_x_ = 5;

  // Add two dimensions to the state vector (nu_a and nu_yawdd)
  n_aug_ = n_x_ + 2;

  // Lambda for mean and covariance calculation
  lambda_ = 3 - n_x_;

  // Predicted sigma points
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // Weights for mean and covariance calculation
  weights_ = VectorXd(2 * n_aug_ + 1);

  // Augmented state vector
  x_aug_ = VectorXd(n_aug_);

  // Augmented process covariance matrix
  P_aug_ = MatrixXd(n_aug_, n_aug_);

  // Augmented sigma points
  Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // Radar measurement covariance matrix
  R_radar_ = MatrixXd(3, 3);
  R_radar_ << std_radr_ * std_radr_, 0.0, 0.0,
              0.0, std_radphi_ * std_radphi_, 0.0,
              0.0, 0.0, std_radrd_ * std_radrd_;

  // Lidar measurement covariance matrix
  R_lidar_ = MatrixXd(2, 2);
  R_lidar_ << std_laspx_ * std_laspx_, 0.0,
              0.0, std_laspy_ * std_laspy_;

  // Initialize the Lidar measurement matrix H_laser
  H_lidar_ = MatrixXd(2, 5);
  H_lidar_ << 1, 0, 0, 0, 0,
              0, 1, 0, 0, 0;

  // Initialize Lidar NIS
  NIS_lidar_ = 0.0;

  // Initialize Radar NIS
  NIS_radar_ = 0.0;

  // Open Lidar NIS file stream
  NIS_lidar_os_.open("nis_lidar.txt");

  // Open Radar NIS file stream
  NIS_radar_os_.open("nis_radar.txt");
}

UKF::~UKF()
{
  NIS_lidar_os_.close();
  NIS_radar_os_.close();
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  double rho, phi, rho_dot, px, py, v, yaw, yawd;
  double delta_t;

  if (!is_initialized_)
  {
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      //range
      rho = meas_package.raw_measurements_[0];
      //bearing
      phi = meas_package.raw_measurements_[1];
      //radial velocity
      rho_dot = meas_package.raw_measurements_[2];
      //convert polar to cartesian
      px = rho * cos(phi);
      if (fabs(px) < 0.001)
        px = 0.001;
      py = rho * sin(phi);
      if (fabs(py) < 0.001)
        py = 0.001;
      v = 0.0;
      yaw = 0.0;
      yawd = 0.0;

    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      //position (x)
      px = meas_package.raw_measurements_[0];
      if (fabs(px) < 0.001)
        px = 0.001;
      //position (y)
      py = meas_package.raw_measurements_[1];
      if (fabs(py) < 0.001)
        py = 0.001;
      //laser does not provide speed measurements
      v = 0.0;
      yaw = 0.0;
      yawd = 0.0;

    }

    //initialize state variable using the first measurement - radar
    x_ << px, py, v, yaw, yawd;

    //update time
    time_us_ = meas_package.timestamp_;

    //done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;

  // Ignore measurement if time delta is too small
  if (delta_t > 0.001)
  {

    Prediction(delta_t);

    if (meas_package.sensor_type_ == MeasurementPackage::LASER)
    {
      UpdateLidar(meas_package);
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
      UpdateRadar(meas_package);
    }
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  double px, py, v, yaw, yawd, nu_a, nu_yawdd;
  double px_next, py_next, v_next, yaw_next, yawd_next;

  /* Generate sigma points */

  // Create the augmented state vector
  // x_aug = [px, py, v, yaw, yawd, nu_a, nu_yawdd] (size: 7)
  x_aug_.head(n_x_) = x_;
  x_aug_(5) = 0.0;
  x_aug_(6) = 0.0;

  // Create the augmented process covariance matrix
  P_aug_.fill(0.0);
  P_aug_.topLeftCorner(5, 5) = P_;
  P_aug_(5, 5) = std_a_ * std_a_;
  P_aug_(6, 6) = std_yawdd_ * std_yawdd_;

  // Create the square root matrix
  MatrixXd L = P_aug_.llt().matrixL();

  // Create augmented sigma points
  Xsig_aug_.col(0) = x_aug_;
  for (int i = 0; i < n_aug_; i++)
  {
    Xsig_aug_.col(i + 1) = x_aug_ + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug_.col(i + 1 + n_aug_) = x_aug_ - sqrt(lambda_ + n_aug_) * L.col(i);
  }

  /* Predict sigma points */

  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    // Extract state variables from the augmented state for better readability
    px = Xsig_aug_(0, i);
    py = Xsig_aug_(1, i);
    v = Xsig_aug_(2, i);
    yaw = Xsig_aug_(3, i);
    yawd = Xsig_aug_(4, i);
    nu_a = Xsig_aug_(5, i);
    nu_yawdd = Xsig_aug_(6, i);

    // Handle division by zero (driving on a straight line)
    if (fabs(yawd) > 0.001) {
        px_next = px + v/yawd * ( sin (yaw + yawd * delta_t) - sin(yaw));
        py_next = py + v/yawd * ( cos(yaw) - cos(yaw + yawd * delta_t) );
    }
    else {
        px_next = px + v * delta_t * cos(yaw);
        py_next = py + v * delta_t * sin(yaw);
    }

    // CTRV model constant velocity
    v_next = v;
    yaw_next = yaw + yawd * delta_t;
    // CTRV model constant acceleration
    yawd_next = yawd;

    // Add noise
    px_next = px_next + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_next = py_next + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_next = v_next + nu_a * delta_t;
    yaw_next = yaw_next + 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_next = yawd_next + nu_yawdd * delta_t;

    // Cache predicted sigma points
    Xsig_pred_(0, i) = px_next;
    Xsig_pred_(1, i) = py_next;
    Xsig_pred_(2, i) = v_next;
    Xsig_pred_(3, i) = yaw_next;
    Xsig_pred_(4, i) = yawd_next;
  }

  /* Predict state mean and state covariance matrix */

  // Set weight
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i = 1; i < 2 * n_aug_ + 1; i++)
    weights_(i) = 0.5/(n_aug_ + lambda_);

  // Predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);

  // Predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // Normalize angles between -PI and PI
    while (x_diff(3)> M_PI) x_diff(3) -= 2.0 * M_PI;
    while (x_diff(3)< -M_PI) x_diff(3) += 2.0 * M_PI;

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /* Lidar update: the Lidar measurement model is linear. As a result a
     Standard Kalman filter is enough to to the update */
  VectorXd z = meas_package.raw_measurements_;
  VectorXd y = z - H_lidar_ * x_;
  MatrixXd Ht = H_lidar_.transpose();
  MatrixXd S = H_lidar_ * P_ * Ht + R_lidar_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;
  // Update state variables after measurement
  x_ = x_ + (K * y);
  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
  P_ = (I - K * H_lidar_) * P_;

  NIS_lidar_ = y.transpose() * Si * y;

  NIS_lidar_os_ << NIS_lidar_ << std::endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /* Radar update */
  double px, py, v, yaw, v1, v2;
  int n_z = 3;

  MatrixXd Zsig = MatrixXd(3, 2 * n_aug_ + 1);

  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    // Extract values from predicted sigma points matrix for better readibility
    px = Xsig_pred_(0, i);
    py = Xsig_pred_(1, i);
    v  = Xsig_pred_(2, i);
    yaw = Xsig_pred_(3, i);

    if (fabs(px) < 0.001)
      px = 0.001;
    if (fabs(py) < 0.001)
      py = 0.001;

    v1 = cos(yaw) * v;
    v2 = sin(yaw) * v;

    // Measurement model
    Zsig(0, i) = sqrt(px * px + py * py);
    Zsig(1, i) = atan2(py, px);
    Zsig(2, i) = (px * v1 + py * v2) / sqrt(px * px + py * py);
  }

  // Mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
      z_pred = z_pred + weights_(i) * Zsig.col(i);

  // Measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    // Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // Normalize angle
    while (z_diff(1)> M_PI) z_diff(1) -= 2.0 * M_PI;
    while (z_diff(1)< -M_PI) z_diff(1) += 2.0 * M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  S = S + R_radar_;

  // Calculate cross correlation matrix
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    // Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // Normalize angle
    while (z_diff(1) > M_PI) z_diff(1) -= 2.0 * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2.0 * M_PI;

    // State difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3) > M_PI) x_diff(3) -= 2.0 * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2.0 * M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // Residual
  VectorXd z = meas_package.raw_measurements_;
  VectorXd z_diff = z - z_pred;

  // Normalize angle
  while (z_diff(1)> M_PI) z_diff(1) -= 2.0 * M_PI;
  while (z_diff(1)<-M_PI) z_diff(1) += 2.0 * M_PI;

  // Update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;

  NIS_radar_os_ << NIS_radar_ << std::endl;
}
