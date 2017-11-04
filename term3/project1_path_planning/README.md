# Path Planning Project
Self-Driving Car Engineer Nanodegree Program

## Background   

In this project a **Path Planning Module** is used for the purpose of navigating around a virtual highway with other traffic that is driving +-10 MPH of the 50 MPH speed limit. The car receives localization and sensor fusion data together with a sparse map list of waypoints around the highway. The car tries to go as close as possible to the 50 MPH speed limit, passing slower traffic when possible. The car avoids hitting other cars and stays inside of the marked road lanes at all times, unless going from one lane to another. The car is able to make one complete loop around the highway. The car does not experience total acceleration over 10 m/s^2 and jerk is never greater than 50 m/s^3.

## Model

The Path Planning Module uses the following classes to accomplish the task of safely driving on the highway:
* Lane (lane.h): this simple class provides a way to distinguish the three different lanes on the highway. There are three lanes, `left`, `center` and `right` starting from the middle of the road.
* State (state.h): this simple class represents the states of the Finite State Machine of the path planning module. The path planner uses a Finite State Machine with four states: `start`, `keep lane`, `change lane left` and `change lane right`.
* Map (map.h and map.cpp): the map is used to load the waypoints on the highway and get the XY coordinates for the estimated
trajectories expressed as Frenet coordinates. The resulting XY coordinates are then sent to the motion control module in charge of executing the meneuver. Splines are used to interpolate the trajectory points and make trajectories smoother.
* Vehicle (vehicle.h and vehicle.cpp): this class describes both the self-driving car driving on the highway and also other vehicles whose data is received through the sensor fusion module. Every time telemetry data is received the status of the self-driving car is updated using the path planning algorithm described later.
* Road (road.h and road.cpp): the road class is used to store vehicles data coming from the sensor fusion module and provides methods helping the planner to decide about driving on the road. The road object has three lanes, each being a vector of vehicle objects, including information about vehicles in each lane of the road.
* Planner (planner.h planner.cpp): the planner class is the class implementing the planning algorithm whose details are given in a dedicated section below.

## Path Planning algorithm

The path planning algorithm implemented in class Planner (planner.h and planner.cpp) is based on a Finite State Machine with four states: `start`, `keep lane`, `change lane left` and `change lane right`. The car will always start in the start state and, depending on the state the car is in at a certain moment, different actions will be performed.

In principle, the car will always try to stay in the same lane (`keep lane state`) and change state only in specific situations. To understand more about how the algorithm is working have a look at `Planner::new_trajectory` method in planner.cpp.
* If the car is in the `start state` it will execute the start action. This state requires specific handling since the car needs to accelerate to reach traffic speed and start moving together with other cars on the highway.
* If the car is in the `keep lane state` it will stay in the same state as long as it is safe to do so (planner.cpp lines 255 - 259). The `Road::is_lane_safe` method returns true or false depending on the distance of vehicles around the self-driving car.
* If it is not safe to stay in `keep lane state` (planner.cpp lines 260 - 275) because, for instance we are approaching another slower vehicle, the car will first try to find a new lane which is safe to move to. If there is such lane which the car can move to safely it will execute the action `change lane` (planner.cpp line 273) otherwise will stay in its lane and slow down to avoid collisions (planner.cpp line 268).
* When the car is in `change lane left state` or `change lane right state` (planner.cpp lines 278 - 291) it will either go to the `keep lane state` or slow down depending on whether the lane is safe or not.

Needless to say that the Finite State Machine is implemented in such a way that if the car is on the right lane moving still to the right is not possible, and the same is true about moving to the left, if the car is in the left lane of the highway.

Moving between states as described above is possible by means of four different actions which are actually implementations of the transition functions of the Finite State Machine: `start`, `keep lane`, `slow down` and `change lane` (planner.cpp lines 332 - 426).

Here is a video showing the self-driving car driving on the highway using the simulator.

[Video](path_planning.mp4)

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
    ## Basic Build Instructions

    1. Clone this repo.
    2. Make a build directory: `mkdir build && cd build`
    3. Compile: `cmake .. && make`
    4. Run it: `./path_planning`.

    Here is the data provided from the Simulator to the C++ Program

### Main car's localization Data (No Noise)

["x"] The car's x position in map coordinates

["y"] The car's y position in map coordinates

["s"] The car's s position in Frenet coordinates

["d"] The car's d position in Frenet coordinates

["yaw"] The car's yaw angle in the map

["speed"] The car's speed in MPH

### Previous path data given to the Planner

["previous_path_x"] The previous list of x points previously given to the simulator

["previous_path_y"] The previous list of y points previously given to the simulator

### Previous path's end s and d values

["end_path_s"] The previous list's last point's frenet s value

["end_path_d"] The previous list's last point's frenet d value

### Sensor Fusion Data, a list of all other car's attributes on the same side of the road. (No Noise)

["sensor_fusion"] A 2d vector of cars and then that car's [car's unique ID, car's x position in map coordinates, car's y position in map coordinates, car's x velocity in m/s, car's y velocity in m/s, car's s position in frenet coordinates, car's d position in frenet coordinates.

## Details

1. The car uses a perfect controller and will visit every (x,y) point it receives in the list every .02 seconds. The units for the (x,y) points are in meters and the spacing of the points determines the speed of the car. The vector going from a point to the next point in the list dictates the angle of the car. Acceleration both in the tangential and normal directions is measured along with the jerk, the rate of change of total acceleration. The (x,y) point paths that the planner receives should not have a total acceleration that goes over 10 m/s^2, also the jerk should not go over 50 m/s^3.

2. There will be some latency between the simulator running and the path planner returning a path, with optimized code usually its not very long maybe just 1-3 time steps. During this delay the simulator will continue using points that it was last given, because of this its a good idea to store the last points you have used so you can have a smooth transition. previous_path_x, and previous_path_y can be helpful for this transition since they show the last points given to the simulator controller with the processed points already removed. You would either return a path that extends this previous path or make sure to create a new path that has a smooth transition with this last path.
