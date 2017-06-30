#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  //initializing matrices
  Hj_ = MatrixXd(3, 4);

  //measurement noise covariance matrix - laser
  R_laser_ = MatrixXd(2, 2);
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement noise covariance matrix - radar
  R_radar_ = MatrixXd(3, 3);
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  //initialize H_laser
  H_laser_ = MatrixXd(2, 4);
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  //state covariance matrix P
  MatrixXd P_ = MatrixXd(4, 4);
  P_ << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;

  //state transition matrix
  MatrixXd F_ = MatrixXd(4, 4);
  F_ << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;

  //process noise covariance marix
  MatrixXd Q_ = MatrixXd(4, 4);

  //state variables (px, py, vx, vy)
  VectorXd x_ = VectorXd(4);

  //initialize the Kalman Filter matrices
  ekf_.Init(x_, P_, F_, H_laser_, R_laser_, Q_);

}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  float px, py, vx, vy, rho, phi, rho_dot, dt, dt2, dt3, dt4;
  float noise_ax = 9.0, noise_ay = 9.0;
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {

    //set state variables (px, py, vx, vy) = (1, 1, 1, 1)
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      //range
      rho = measurement_pack.raw_measurements_[0];
      //bearing
      phi = measurement_pack.raw_measurements_[1];
      //radial velocity
      rho_dot = measurement_pack.raw_measurements_[2];
      //convert polar to cartesian
      px = rho * cos(phi);
      py = rho * sin(phi);
      vx = rho_dot * cos(phi);
      vy = rho_dot * sin(phi);
      //initialize state variable using the first measurement - radar
      ekf_.x_ << px, py, vx, vy;
    } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      //position (x)
      px = measurement_pack.raw_measurements_[0];
      //position (y)
      py = measurement_pack.raw_measurements_[1];
      //laser does not provide speed measurements
      vx = 0.0;
      vy = 0.0;
      //initialize state variable using the first measurement - laser
      ekf_.x_ << px, py, vx, vy;
    }

    //update time
    previous_timestamp_ = measurement_pack.timestamp_;

    //done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

 //compute time delta
 dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
 dt2 = dt * dt;
 dt3 = dt2 * dt;
 dt4 = dt3 * dt;

 //update state transition matrix to account for time delta (dt)
 ekf_.F_ = MatrixXd(4, 4);
 ekf_.F_ << 1, 0, dt, 0,
            0, 1, 0, dt,
            0, 0, 1, 0,
            0, 0, 0, 1;

  //update the process noise covariance marix
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ <<  dt4 / 4.0 * noise_ax, 0.0, dt3 / 2.0 * noise_ax, 0.0,
              0.0, dt4 / 4.0 * noise_ay, 0.0, dt3 / 2.0 * noise_ay,
              dt3 / 2.0 * noise_ax, 0.0, dt2 * noise_ax, 0.0,
              0.0, dt3 / 2.0 * noise_ay, 0.0, dt2 * noise_ay;

  //predict (px, py, vx, vy) using the Kalman Filter motion model equaton
  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    //compute the Jacobian matrix
    Hj_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.H_ = Hj_;
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  previous_timestamp_ = measurement_pack.timestamp_;
}
