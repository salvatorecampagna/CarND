# PID Controller Project
Self-Driving Car Engineer Nanodegree Program

## Background   

In this project a **PID controller** is used for the purpose of controlling the steering of a self-driving car.
The car is driven inside a simulator which provides the Cross-Track Error (CTE), the speed and the angle as input for the PID controller. The PID controller has to provide as output the steering command to drive the car on the track inside the simulator.  

A PID Controller uses a Proportional (P), a Differential (D) and an Integral (I) component to compute the output command. Each component contirbutes to the output according to a gain parameter, one for each component, Kp, Ki and Kd. Each of these three components serves a specific purpose and has a different effect on the output of the PID controller.  

The effect of each component is as follows:

* Proportional (P): it contributes to steering the car by a value that is proportional to the Cross-Track Error (CTE) that the PID reads as input.
* Differential (D): contributes to smoothing the overshoot effect typical of a P controller.
* Integram (I): contributes to correcting bias errors resulting in a PD controller not being able to reach a correct steady state value.  


Coding a PID controller in C++ is quite easy to do as you can see having a look at the source code in `PID.h` and `PID.cpp`. Much harder is to find the right set of parameters for the PID controller. A PID controller needs three parameters to be tuned to have good performances:  

* Kp: proportional gain (weight for the proportional component)
* Ki: integral gain (weight for the integral component)
* Kd: differential gain (weight for the differential component)

Most of the time devoted to this project has been spent finding the right combination of values for these parameters. I used a trial and error approach to find good enough values and also some heuristics including:  

* Keeping the proportional gain low to favor stability of the PID controller. Increasing the proportional gain usually makes the PID controller unstable.

* Use the integral gain to adjust offset errors. This is easy to spot since it is visible by looking if the car stays close to the center line of the track (after a turn).

* Use the differential gain to adjust for delay errors and control overshoots. The derivative gain value is important to control overshoots and is useful to stabilize the output.

After some tuning, the set of parameters found good enough to drive the car in the simulator are the following:  

* Kp: 0.13
* Ki: 0.0002
* Kd: 3.0

## Dependencies  

* cmake >= 3.5
 * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `./install-mac.sh` or `./install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```
    Some function signatures have changed in v0.14.x. See [this PR](https://github.com/udacity/CarND-MPC-Project/pull/3) for more details.
* Simulator available [here](https://github.com/udacity/self-driving-car-sim/releases) in the classroom.

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./pid`. 

The `pid` executable accepts 3 parameters for `Kp`, `Ki` and `Kd`. To use the parameters I have found run the PID controller as follows:  

`./pid 0.13 0.0002 3.0`  

The source code provided in `main.cpp` implements the communication protocol with the simulator and calls the PID controller methods to inizialize and compute the control command. After the PID contoller is running the simulator can be started and will listen for PID controller commands on a web socket. To simulate the car driving under control of the PID controller select "Project 4: PID Controller" in the simulator. As a result the car will start moving receiving control command from the PID controller.