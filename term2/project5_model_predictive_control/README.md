# Model Predictive Control Project
Self-Driving Car Engineer Nanodegree Program

[state_equations]: ./images/state.png "Vehicle state equations"
[error_equations]: ./images/errors.png "Cross-Track and Heading Errors"

## Background   

In this project a **Model Predictive Controller (MPC)** is used for the purpose of controlling the steering and acceleration of a self-driving car. The car is driven inside a simulator. The MPC has to provide as output the steering and acceleration commands to drive the car on the track in the simulator.  

A Model Predictive Controller (MPC) is an advanced method for controlling systems whose behavior is described by a dynamic model. Dynamic models describe the actual vehicle dynamics as closely as possible so to include also tyre forces, longitudinal and lateral forces, inertia, gravity, air resistence, drag, mass and geometry of the vehicle.  

In this project the vehicle behavior is modeled by means of a simplified, but good enough, **kinematic model**. The kinematic model ignores things like gravity, mass and tyre forces amd is relatively simple to use.  

## Model

The kinematic model of the vehicle captures how the the state of the vehicle evolves over time and the control inputs required to change it. Controlling of the vehicle state takes place by means of actuator control inputs.
Usually, cars have three actuators: the steering wheel, the throttle pedal and the brake pedal. In this case, anyway, we consider the throttle and brake pedal as a single actuator with negative values meaning breaking and positive values meaning accelerating. Actually, in our case, the throttle control will have values in the range [-1, 1] with -1 meaning full brake and 1 meaning full acceleration.  

As a result of this simplification the number of control inputs for the system is reduced to two: the steering angle control input and the acceleration control input.  

The kinematic model describing the vehicle dynamics is the following:

![alt text][state_equations]

The state vector used to describe the state of the car is the following: `[x, y, psi, v]`, where:
* `x`: x position of the car
* `y`: y position of the car
* `psi`: heading of the car
* `v`: velocity of the car

The actuator control inputs include: `[delta, a]`, where:
* `delta`: steering angle control input [-25, 25] degrees
* `a`: acceleration control input [-1,1]

Here `Lf` is a parameter measuring the distance between the front of the vehicle and the center of gravity (CoG) of the car.  

Usually autonomous vehicle system architectures include a **path planning** module passing a **reference trajectory** that the vehicle has to follow. This reference trajectory could be represented by means of polynomials, with third degree polynomials being common. A control algorithm, such as MPC, is in charge of applying control inputs `(delta, a)` in such a way that the vehicle follows the reference trajectory.  

Once the reference trajectory is available, the vehicle kinematic model is used to predict the vehicle trajectory. At this point the controller has to act on control inputs `(delta, a)` to minimize the difference between the predicted (by the kinematic model) and reference (provided by the path planning module) trajectory. In summary, we would like to find suitable values for `(delta, a)` such that the difference between the predicted trajectory and reference trajectory is minimized. Additionally, we would like to minimize also the difference between the vehicle orientation and the trajectory orientation (heading).  

In order to take into account quantities we want to minimize, we extend our state vector to include also:
* cte: Cross-Track Error, being the distance between the reference trajectory and the predicted trajectory
* epsi: heading error, being the difference between the vehicle orientation and the trajectory orientation

The Cross-Track Erro and Heading Error equations are as follows:

![alt text][error_equations]

At this point we have a complete kinematic model of the car including state variables, control inputs, Cross-Track Error and Heading Error.  

We are still missing a piece anyway, a way to minimize the Cross-Track Error (cte) and Heading Error (epsi) considering that we can only act on our car by means of the control inputs. Moreover, we would like to put some constraints on the values of `(delta, a)` to enforce both physical constraints, such as finite acceleration and steering angle, and smoothness of steering and acceleration. For this reason we define a cost function (`MPC.cpp` line 59-82) that captures the errors we want to minimize (cte and epsi) and the constraints. The cost will be minimized in such a way to find the values of actuator control inputs `(delta, a)` providing minimum value for the cost function itself. As a result, we will find steering angle and acceleration actuator control inpput values minimizing the Cross-Track Error (cte) and the Heading Error (epsi).


## Timeslot and horizon

One of the main advantages of MPC over other control algorithms is represented by the possibility of tuning two parameters like the timeslot and horizon `(dt, N)`. The idea behind this two parameters is to fit a a predicted trajectory matching as close as possible a reference trajectory by using N points spaced by dt seconds. As a result of this operation, (N-1) control inputs are calculated which minimize the Cross-Track Error and Heading Error. This way we can optimize a finite time horizon to control our car paying attention to only using the first of the (N-1) control commands to command the steering angle and acceleration of the car and discarding the remaining. For the following time steps the procedure is repeted computing (N-1) control inputs and using only the first one again.    
Tuning the values of N and dt is a matter of trading-off since their product provides the amount of time in future we woud like to predict the bahvior of the vehicle but also how often we control the vehicle itself. Smaller values for dt result in a finer grained control of the car with control inpts fed more frequently. Acting on N, instead, we decide for how long we are trying to predict the vehcile behavior, or put another way, how much reference trajectory we would like to fit. Also we have to consider that having many points (high values for N) and shorter timeslots, increases the computational cost of running the solver to find the values of the steering angle and delta.  

For this project different values of dt and N have been tried, including the following:  

* dt = 0.05, N = 20 (1 second horizon)
* dt = 0.05, N = 15 (0.75 seconds horizon)
* dt = 0.05, N = 10 (0.5 seconds horizon)
* dt = 0.1, N = 10 (1 second horizon)
* dt = 0.1, N = 12 (1.2 seconds horizon)
* dt = 0.1, N = 15 (1.5 seconds horizon)
* dt = 0.1, N = 20 (2 seconds horizon)

At the end the values chosen are `dt = 0.1` and `N = 10` and the choice has been taken by looking at the bahavior of the car in the simulator. With these values the car is reactive enough but also steering and acceleration are smooth (this is also due to tuning of the cost function used).


## Waypoint pre-processing, polyfit and control

In `main.cpp` at lines 129-144 waypoints are pre-processed to convert the position of the car from the map coordinate system to vehicle coordinate system. A set of equations for rotation and translation is used. At line 148 the coefficients of the line fitting the reference trajectory are computed while at line 151 and 152 the Crss-Track Error and the Heading Error are computed. At line 160 the state vector is created with x, y and psi being 0 as a result of using the vehicle reference system. Then, at line 176, the state and polynomial fit are passed to the solver that returns the vector of control inputs. At line 184 and 185 the control inputs returned (just the first two) are used to feed the simulator with the streering angle and acceleration. The steering angle sign is changed to account for turning left for counterclockwise angles and turning right for clockwise angles.

## Latency handling

Having a close look at the code in `main.cpp`, line 216, a delay is introduced before actuating the control inputs of the car. This simulates the delay taken for the actual actuation of the steering and acceleration control inputs. We need to take care of this delay since, failing to take into account this latency would result in the control commands to be applied in a future time, when the state of the vehicle is different by the state the vehicle is when the control input is computed. To handle this situation at line 172 the future state of the vehicle is estimated. If we consider a constant delay of 100 milliseconds (as specified at line 216) what we do, at line 172, is to use the kinematic model of th car to estimate the state of the car 100 ms ahead of the current time. This results in computing control inputs for `t + 100 ms` instead of `t`. At the end when the command reaches the actuators and is actuated the vehicle state predicted by the kinematic model is very close (not exact since the kinematic model does not capture all the details like a dynamic model) to the real vehicle state and the control input is applied to a state that is very close to the state of the vehicle.  

 ## Cost function

 In `MPC.cpp`, lines 59-82 the cost function minimized by our solver is available. The solver will solve the set of equations trying to minimize this function and provide the values of `delta` and `a` providing the minimum cost. At line 61 and 63 the cost function includes the CrossTrack and Heading error with hih values as weighst sice we really want our solver minimize as much as possible the distance of the predicted trajectory from the reference trajectory and the heading error. At line 65 we add a term which prevents the car from stopping and, moreover, to keep the reference velocity of 100 mph. At line 71 and 73 we also add the steering angle and acceleration to the cost doing something similar to what has been done before but with lower values for the weights in front. At line 79 and 81 we also add the difference between the current and previous control inputs to the cost in such a way to have smoother steering and acceleration.

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
  * Run either `install-mac.sh` or `install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```
    Some function signatures have changed in v0.14.x. See [this PR](https://github.com/udacity/CarND-MPC-Project/pull/3) for more details.
* Fortran Compiler
  * Mac: `brew install gcc` (might not be required)
  * Linux: `sudo apt-get install gfortran`. Additionall you have also have to install gcc and g++, `sudo apt-get install gcc g++`. Look in [this Dockerfile](https://github.com/udacity/CarND-MPC-Quizzes/blob/master/Dockerfile) for more info.
* [Ipopt](https://projects.coin-or.org/Ipopt)
  * Mac: `brew install ipopt`
       +  Some Mac users have experienced the following error:
       ```
       Listening to port 4567
       Connected!!!
       mpc(4561,0x7ffff1eed3c0) malloc: *** error for object 0x7f911e007600: incorrect checksum for freed object
       - object was probably modified after being freed.
       *** set a breakpoint in malloc_error_break to debug
       ```
       This error has been resolved by updrading ipopt with
       ```brew upgrade ipopt --with-openblas```
       per this [forum post](https://discussions.udacity.com/t/incorrect-checksum-for-freed-object/313433/19).
  * Linux
    * You will need a version of Ipopt 3.12.1 or higher. The version available through `apt-get` is 3.11.x. If you can get that version to work great but if not there's a script `install_ipopt.sh` that will install Ipopt. You just need to download the source from the Ipopt [releases page](https://www.coin-or.org/download/source/Ipopt/).
    * Then call `install_ipopt.sh` with the source directory as the first argument, ex: `sudo bash install_ipopt.sh Ipopt-3.12.1`. 
  * Windows: TODO. If you can use the Linux subsystem and follow the Linux instructions.
* [CppAD](https://www.coin-or.org/CppAD/)
  * Mac: `brew install cppad`
  * Linux `sudo apt-get install cppad` or equivalent.
  * Windows: TODO. If you can use the Linux subsystem and follow the Linux instructions.
* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page). This is already part of the repo so you shouldn't have to worry about it.
* Simulator. You can download these from the [releases tab](https://github.com/udacity/self-driving-car-sim/releases).
* Not a dependency but read the [DATA.md](./DATA.md) for a description of the data sent back from the simulator.


## Basic Build Instructions


1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./mpc`.
