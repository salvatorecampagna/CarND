# Unscented Kalman Filter Project
Self-Driving Car Engineer Nanodegree Program

[lidar_nis_dataset1]: ./images/lidar_nis_dataset1.jpg "NIS Lidar Dataset 1"
[radar_nis_dataset1]: ./images/radar_nis_dataset1.jpg "NIS Radar Dataset 1"
[lidar_nis_dataset2]: ./images/lidar_nis_dataset2.jpg "NIS Lidar Dataset 2"
[radar_nis_dataset2]: ./images/radar_nis_dataset2.jpg "NIS Radar Dataset 2"
[radar_nis_only_radar_dataset1]: ./images/radar_nis_only_radar_dataset1.jpg "NIS Radar only Dataset 1"
[radar_nis_only_radar_dataset2]: ./images/radar_nis_only_radar_dataset2.jpg "NIS Radar only Dataset 2"
[lidar_nis_only_lidar_dataset1]: ./images/lidar_nis_only_lidar_dataset1.jpg "NIS Lidar only Dataset 1"
[lidar_nis_only_lidar_dataset2]: ./images/lidar_nis_only_lidar_dataset2.jpg "NIS Lidar only Dataset 2"

# Background

The Kalman Filter is an algorithm that uses a series of measurements affected by noise to estimate a set of unknown variables describing the state of a system. It uses a continuous loop of `prediction`/`update` steps to estimate the state of the object by updating the estimation of a set of state variables using measurements coming from sensors. The prediction step is described by a (possibly non-linear) `process model`, while the update step is described by a (possibly non-linear) `measurement model`.

If the process and/or the measurement models are linear a Standard Kalman Filter can be used to estimate the state of the system. If the process model and/or the measurement model are described by non-linear equations an Extended Kalman Filter or an Unscented Kalman Filter is required.

The Unscented Kalman Filter is an alternative technique, with respect to an Extended Kalman Filter, to deal with non-linear process and/or measurement models.
The main difference between an Extended Kalman Filter and an Unscented Kalman Filter is represented by the way they handle linearization of a non-linear process and/or measurement model. The Extended Kalman Filter uses multivariate Taylor Series expansions to linearize a model about a working point. The Unscented Kalman Filter, instead, uses a so called `Unscented Transformation` which in turn uses a set of sigma points to approximate a probability distribution. It turns out that sigma points provide an approximation that is better than a linearization, which explains why in practical applications an Unscented Kalman Filter outperforms an Extended Kalamn Filter.

In this project an Unscented Kalman Filter is used to estimate the state of a moving object of interest with noisy Lidar and Radar measurements.
Moreover, the process model used, describes the motion model of the object to track using the so called `Constant Turn Rate and Velocity magnitude model` (CTRV). This process model is a good process model to describe a vehicle behavior in real traffic scenarios. The CTRV model is described by a non-linear model and, as a result, requires an algorithm such as the Unscented Kalman Filter to have a good estimation of the tracked object state.

The state of the object we want to track is described by a state vector including 5 variables:
* Position X
* Position Y
* Velocity (magnitude)
* Yaw
* Yaw rate

Further, the uncertainty of the process model is described by another 2 dimensional noise vector consisting of two independent scalar noise processes. The first is the `longitudinal acceleration noise`, the second is the `yaw acceleration noise`. Both of them are normally distributed white noise with mean zero.

## Algorithm

The Unscented Kalman Filter estimates the state of the tracked object going through two steps, each having different substeps:
* Prediction
* Update

Prediction:

The prediction step goes through the following three steps:
* Sigma points generation
* Sigma points prediction
* Process mean and covariance matrix prediction

Update:

The update step goes through the following three steps:
* Prediction measurement
* State update

At the end of each `Prediction`/`Update` cycle the Unscented Kalman Filter provides as output a state vector including the mean value for all the estimated state variables and a covariance matrix describing the uncertainty in the estimation of each variable.

## Remarks

The Unscented Kalman Filter has a set of properties that make it suitble to be used as a sensor fusion tool for Self-Driving Cars. The three most important properties of the Unscented Kalman Filter are:

1. An Unscented Kalamn Filter can use noisy measurement data as input to accurately estimate position and velocity of an object without introducing significant delays;

2. An Unscented Kalman Filter can estimate the orientation (yaw) and yaw rate of the object using sensors (such as a Radar and a Lidar) that can't even directly observe orientation and yaw rate;

3. The Unscented Kalman Filter, by means of a covariance matrix, gives also information on how precise the estimation is. Moreover, a consistency check of the filter allows to know if the uncertainty described by the covariance matrix is realistic or not. Such consistency check is done on the Normalized Innovation Squared (NIS) value by means of a `Chi-Squared` test;

# Project description

This project requires the Term 2 Simulator which can be downloaded [here](https://github.com/udacity/self-driving-car-sim/releases)

Please install uWebSocketIO to support communication between the simulator and the Kalman Filter.
Use install-ubuntu.sh on Linux or Windows/Docker, or install-mac.sh for Mac.

Once the install for uWebSocketIO is complete, the main program can be built and run by doing the following from the project top directory.

1. mkdir build
2. cd build
3. cmake ..
4. make
5. ./ExtendedKF

The Unscented Kalman Filter implementation is available in `src/ukf.cpp` and `src/ukf.h`.
Additional utility functions are provided in `tools.cpp` and `tools.h`.
The program `main.cpp` implements the main logic to feed the Unscented Kalman Filter with data and collect results.

Here is the main protocol that `main.cpp` uses for uWebSocketIO in communicating with the simulator.

INPUT: values provided by the simulator to the C++ program implementing the Unscented Kalman Filter

* ["sensor_measurement"] => the measurement that the simulator observed (either LIDAR or RADAR)

OUTPUT: values provided by the C++ program to the simulator

* ["estimate_x"] <= Kalman filter estimated position x
* ["estimate_y"] <= Kalman filter estimated position y
* ["rmse_x"] <= Root Mean Squared error on px
* ["rmse_y"] <= Root Mean Squared error on py
* ["rmse_vx"] <= Root Mean Squared error on vx
* ["rmse_vy"] <= Root Mean Squared error on vy

RMSE values measure the accuracy of the filter in estimating the position and velocity of the object. The values are derived by comparing the estimation provided by the filter against the ground thruth data available by means of simulated Lidar and Radar data.
---

## Project results

The performance of the Unscented Kalman Filter designed in this project are evaluated by comparing the estimations provided by the flter against ground truth data. The metric used to measure the accuracy of the filter estimation is the Root Mean Squared Error (RMSE).
Moreover, two datasets including noisy Lidar and Radar data are used.

The performances of the filter have been evaulated using both the Lidar and Radar data first.

In this scenario RMSE values are the following:
* Dataset 1: [0.0710, 0.0857, 0.3745, 0.2995]
* Dataset 2: [0.0715, 0.0700, 0.3629, 0.2353]

Consistency check on NIS value on both Lidar and Radar measurement space on both datasets confirm the filer is well conditioned and that the estimation is, as a result, realistic.

![alt text][lidar_nis_dataset1]
![alt text][radar_nis_dataset1]
![alt text][lidar_nis_dataset2]
![alt text][radar_nis_dataset2]

If we try to use only the Lidar we obtain the following results:

* Dataset 1: [0.1037, 0.0997, 0.5518, 0.3621]
* Dataset 2: [0.0906, 0.0912, 0.5594, 0.4822]

Again, consistency on NIS value confirms realistic estimations.

![alt text][lidar_nis_only_lidar_dataset1]
![alt text][lidar_nis_only_lidar_dataset2]

If we discard Lidar data and instead use Radar data only the results are the following:

* Dataset 1: [0.1657, 0.2488, 0.5499, 0.3515]
* Dataset 2: [0.2944, 0.6520, 0.1533, 0.1928]

Consistency on NIS value confirms realistic estimations.

![alt text][radar_nis_only_radar_dataset1]
![alt text][radar_nis_only_radar_dataset2]


## Other Important Dependencies

* cmake >= 3.5
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
   * On windows, you may need to run: `cmake .. -G "Unix Makefiles" && make`
4. Run it: `./UnscentedKF`

## Simulation

1. Run the simulator (see above for download link)
2. Select 'Project 1/2: EKF and UKF'
3. Select Dataset1 or Dataset2 (each dataset includes LIDAR and RADAR measurements)
4. Hit 'Start' button to start Simulation
5. Observe RMSE on X, Y, VX and VY
