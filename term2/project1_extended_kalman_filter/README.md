# Extended Kalman Filter Project Starter Code
Self-Driving Car Engineer Nanodegree Program

In this project a kalman filter is used to estimate the state of a moving object of interest with noisy LIDAR and RADAR measurements.

This project requires the Term 2 Simulator which can be downloaded [here](https://github.com/udacity/self-driving-car-sim/releases)

Please install uWebSocketIO to support communication between the simulator and the Kalman Filter.
Use install-ubuntu.sh on Linux or Windows/Docker, or install-mac.sh for Mac.

Once the install for uWebSocketIO is complete, the main program can be built and run by doing the following from the project top directory.

1. mkdir build
2. cd build
3. cmake ..
4. make
5. ./ExtendedKF

Implementation of the sensor fusion logic is available in: src/FusionEKF.cpp, src/FusionEKF.h.
The Kalman Filter implementation is available in: kalman_filter.cpp, kalman_filter.h.
Additional utility functions are provided through: tools.cpp, and tools.h

The program main.cpp implements the main logic to feed the Kalman Filter with data and collect results.

Here is the main protcol that main.cpp uses for uWebSocketIO in communicating with the simulator.

INPUT: values provided by the simulator to the C++ program

["sensor_measurement"] => the measurement that the simulator observed (either LIDAR or RADAR)

OUTPUT: values provided by the CC++ program to the simulator

["estimate_x"] <= kalman filter estimated position x
["estimate_y"] <= kalman filter estimated position y
["rmse_x"] <= Root Mean Squared error on px
["rmse_y"] <= Root mean squared error on py
["rmse_vx"] <= Root mean squared error on vx
["rmse_vy"] <= Root mean squared error on vy

---

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

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
   * On windows, you may need to run: `cmake .. -G "Unix Makefiles" && make`
4. Run it: `./ExtendedKF `
