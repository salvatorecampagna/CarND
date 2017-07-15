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

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

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
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  double rho, phi, rho_dot, px, py, v, yaw, yawd;

  if (!is_initialized_)
  {
    x_ << 1.0, 1.0, 1.0, 1.0, 1.0;

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      //range
      rho = meas_package.raw_measurements_[0];
      //bearing
      phi = meas_package.raw_measurements_[1];
      //radial velocity
      rho_dot = meas_package.raw_measurements_[2];
      //convert polar to cartesian
      px = rho * cos(phi);
      py = rho * sin(phi);
      v = 0.0;
      yaw = 0.0;
      yawd = 0.0;

    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      //position (x)
      px = meas_package.raw_measurements_[0];
      //position (y)
      py = meas_package.raw_measurements_[1];
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


}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
}
