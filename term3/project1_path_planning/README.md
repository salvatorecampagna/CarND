# Path Planning Project
Self-Driving Car Engineer Nanodegree Program

## Background   

In this project a **Path Planning Module** is used for the purpose of navigating around a virtual highway with other traffic that is driving +-10 MPH of the 50 MPH speed limit. The car receives localization and sensor fusion data together with a sparse map list of waypoints around the highway. The car tries to go as close as possible to the 50 MPH speed limit, passing slower traffic when possible. The car tries to avoid hitting other cars and stay inside of the marked road lanes, unless going from one lane to another. The car is able to make one complete loop around the highway. The car does not experience total acceleration over 10 m/s^2 and jerk is never greater than 50 m/s^3.

## Model

The Path Planning Module uses the following classes to accomplish the task of safely driving on the highway:
* Lane (lane.h): this simple class provides a way to distinguish the three different lanes on the highway. There are three lanes, `left`, `center` and `right` starting from the middle of the road.
* State (state.h): this simple class represents the states of the Finite State Machine of the path planning module. The path planner uses a Finite State Machine with four states: `start`, `keep lane`, `change lane left` and `change lane right`.
* Map (map.h and map.cpp): the map is used to load the waypoints on the highway and get the XY coordinates for the estimated
trajectories expressed as Frenet coordinates. The resulting XY coordinates are then sent to the motion control module in charge of executing the meneuver. Splines are used to interpolate the trajectory points and make trajectories smoother together with 5 degree polynomials which are used to generate Jerk Minimum trajectories.
* Vehicle (vehicle.h and vehicle.cpp): this class describes both the self-driving car driving on the highway and also other vehicles whose data is received through the sensor fusion module. Every time telemetry data is received the status of the self-driving car is updated using the path planning algorithm described later.
* Road (road.h and road.cpp): the road class is used to store vehicles data coming from the sensor fusion module and provides methods helping the planner to decide about driving on the road. The road object has three lanes, each being a vector of `Vehicle` objects.
* Planner (planner.h planner.cpp): the planner class is the class implementing the planning algorithm whose details are given in a dedicated section below.

## Path Planning algorithm

The path planning algorithm implemented in class Planner (planner.h and planner.cpp) is based on a Finite State Machine with four states: `start`, `keep lane`, `change lane left` and `change lane right`. The car will always start in the start state and, depending on the state the car is in at a certain moment, different actions will be performed.

In principle, the car will always try to stay in the same lane (`keep lane state`) and change state only in specific situations such as approaching a slower vehicle on the same lane. To understand more about how the algorithm is working have a look at `Planner::new_trajectory` method in planner.cpp (lines 135 - 196).

* If the car is in the `start state` it will execute the start action (planner.cpp, lines 145 - 148). This state requires specific handling since the car needs to accelerate to reach traffic speed and start moving together with other cars on the highway. The action performed while in the stat state (planner.cpp, lines 277 - 302) will use the speed of the vehicle driving in front of our car to compute the target velocity, which, in turn, is used to compute the target value of s (planner.cpp, lines 287 - 289). These values are used, later (planner.cpp, lines 291 - 294) to compute the start and end points for the s(t) and d(t) polynomials representing the s and d Frenet coordinates of the line the car will follow during the start state.

* If the car is in the `keep lane state` it will stay in the same state as long as it is safe to do so (planner.cpp lines 150 - 173). The `Road::is_lane_safe` method (road.cpp, lines 79 - 97) returns true or false depending on the distance of vehicles around the car and is used to determine if the car should slow down or change lane while staying in the same lane. If there is enough space the car will keep on staying in the same lane otherwise will either try to slow down or change lane. To be more precise, if the planner decides that it is not safe to stay in the same lane it will invoke `Road::find_target_lane` (planner.cpp, line 161) which is in charge of finding a new lane to move to. Details about how a new lane is found are given in a dedicated section below. If invoking the `Road::find_target_lane` returns the current lane as the lane to move to it means we are already in the best lane which anyway is unsafe, as a result the car will invoke `Planner::slow_down`. If the lane returned by `Road::find_target_lane` is different then the planner will call `Road::change_lane` to move to the new lane.

* When the car is in `change lane left state` or `change lane right state` (planner.cpp lines 175 - 188) it will either go to the `keep lane state` or slow down depending on whether the lane is safe or not, which will be determined again calling `Road::is_lane_safe`.

Needless to say that the Finite State Machine is implemented in such a way that if the car is on the right lane moving still to the right is not possible, and the same is true about moving to the left, if the car is in the left lane of the highway.

Moving between states as described above is possible by means of four different actions which are actually implementations of the transition functions of the Finite State Machine: `start`, `keep lane`, `slow down` and `change lane` (planner.cpp lines 277 - 396).

## Changing lane

Before being able to perform a change lane action the planner needs to determine which lane to move to. The selection of the lane to move to is performed by the function `Road::find_target_lane` (road.cpp, line 141 - 193) which uses `Road::find_low_cost_lane`. This function uses a set of cost functions to determine the cost for each lane and returns the index of the lane whose cost is lower (0: left lane, 1: center lane, 2: right lane). This implementation of the planner uses three cost functions to determine the cost of each lane:

1. `Road::lane_cars_cost`: this cost function returns high cost for lanes which are occupied by more cars and lower cost for free lanes. The idea is to favor free lanes to move to, since, in principle, the chances to hit another car in a free lane are lower.

2. `Road::lane_speed_cost`: this cost function returns high cost for lanes whose average speed of cars is lower and low cost for lanes whose average speed is higher. Using this cost function it is possible to consider switching to lanes whose cars are moving faster on average.

3. `Road::lane_distance_cost`: this cost function finds the closest car for each lane and returns a lower cost for the lane whose closest car is further and higher cost for the lane whose closest car is closest. Using this cost function it is possible to consider switching to lanes whose closest car in the lane is farther away. This way switching to lanes with more space in front of the car is favored.

These three cost functions are combined in a single cost function by function `Road::lane_cost` (road.cpp, lines 293 - 305) and different weights are used to balance the *importance* of each cost function. params.h includes the weights used for each cost function.

When it comes to finding the right lane to move to another important function is `Road::find_target_lane` (road.cpp, lines 141 - 193). This function makes sure that, once the lane has been selected using the cost functions mentioned above, the transition is actually doable. At first (road.cpp, lines 153 - 168) it handles two specific cases:

1. The car is in the left lane but the best lane to move to is the right lane. No direct left to right lane is allowed, it would be dangerous so a transition through the center lane is performed (road.cpp, line 158).

2. Similarly if the car is in the right lane but the best lane to move to is the left lane, the car goes through the center lane (road.cpp, line 166).

After that the other two cases are trivial, and handle the situation in which the car is in the left or right lane and the best lane to move to is the center lane or, if the car is in the center lane and the best lane to move to is the left or right lane (road.cpp, lines 171 - 190).

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
