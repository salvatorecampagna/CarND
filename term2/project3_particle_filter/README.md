# Particle Filter Project
Self-Driving Car Engineer Nanodegree Program

In this project a **Particle Filter** is used for the purpose of localizing a car given a set of noisy LIDAR measurements and a 2D map.
The Particle Filter is implmented as a class in `particle_filter.h` and `particle_filter.cpp`. 

The car has been kidnapped and transported to a new location. As a result, no information about the location
is avilable to the car excluding a noisy GPS estimation. The car also has a map of the location and can rely
on lots of noisy LIDAR measurements and control data. 

The Particle Filter implemented in this project estimates the position of the car in a 2D environment.

# Background

A Particle Filter is a filter used to estimate the internal state of a dynamic system when partial observations are made and random perturbations are present in the sensors as well as in the dynamic system. The Particle Filter computes the conditional probability (also posterior distribution) of the states of some Markov process, given some noisy and partial observations.
In this project we try to solve a localization problem and, as a result, the Particle Filter is used to estimate the position of the car in a 2D world (x,y) and its heading.

Among the algorithms used to estimate the state of a system it is the easiest to program and also the most flexible. It works with a continuous state space estimating a multimodal posterior distribution. However, Particle Filters do not perform well when applied to very high-dimensional systems.
In this project, anyway, it is used to estimate the position (x,y) of a car in a 2D space and its heading, so to a 3-dimentional estimation problem, which makes the Particle Filter quite efficient and suitable for real-time processing.

The idea behind a Particle Filter is to use a set of particles each being a discrete guess about the position of an object, in this case of a car. Each single guess includes a position (x,y) and an angle (theta), the heading of the car. The set of several such guesses make the Particle Filter and together approximate a posterior probability for the car position.

The Particle Filter can rely also on a map, in this case a landmark-based map, and on measurements to compute the posterior distribution.

## Particle Filter algorithm steps

1. Initialization

To accomplish the purpose of estimating the posterior distribution of the car position a Particle Filter uses a set of particles. Once the particles are created they must be initialized with an initial position and heading. For the purpose of this project GPS coordinates are fed to the Particle Filter initialization procedure. Samples are drawn from a Gaussian distribution centered around the position provided by the GPS to take into account for sensor noise around the initial GPS position. The result is a set o particles whose position and heading is centered around the GPS coordinates. Note that the number of particles to use is a tuning parameter and experiments are required to select the right number of particles. For applications requiring real-time performances having fewer particles plays an important role with regard to the time spent in prcessing each particle.
Theoretcally, anyway, as the number of particles used approaches infinity the posterior distribution estimated by the Partcle Filter approaches a Bayes posterior ditribution.

2. Prediction step

In this step the algorithm predicts the position and heading of the car at the next time step. For each particle the algorithm updates the particle's location and heading based on the velocity and yaw rate measurements coming from the sensors. To be more precise, the equations used for the prediction step are the equations describing a **CTRV** (Constant Turn Rate and Velocity) motion model. To account for the uncertainty in the control input Gaussian noise is added to the velocity and yaw rate.

3. Data transformation and association

Before we can use landmark measurements of the objects around us to update the posterior probability (belief of the car position and heading) we need to solve the so called **data association problem**. This problem descibes the problem of matching objects in real world, whose position is available by means of a map, to landmark measurements.
One way to address and solve the data association problem is to pick and match leandmarks with measurements using a **nearest neighbor algorithm**. The idea behind this approach, is to match the closest measurement as a correct match for landmarks in the map.

4. Update step

At this stage the measurements come into play. Instead of affecting directly the prediction of the state of the car, measurements are used to determine the **weight** of each particle. Intuitively, the weight of a particle is a measure of importance of that particle, or put it another way, a measure of how much that particle represents the estimated position of the car. In this project the weight of each particle is updated using a **Multivariate Gaussian probability density function** for each measurement. So, each particle is used to compute a partial weight against each measurement (using the Gaussian) and then all this partial weights are multiplied to give the weight for the particle. This procedure needs to be repeted for all particles.

5. Resampling

Once each particle's weight has been updated a resample procedure takes place to select the particles used in the next time step. The resampling procedure is driven by the idea of selecting particles to keep in the next iteration, with a probability proportional to each particle's weight.
As a result of this resampling strategy, particles whose weight is higher have a higher likelihood of `surviving` the sampling procedure and ending up in the set of particles to use in the next step. Be carefult, the resampling procedure is a resampling with repetion, which means each particle could be picked multiple times and end up in the set of particles to use in the next iteration more than once. This suggests also that particles whose weight is higher have a higher probability of being selected multiple times during resampling, while particles whose weight is low are very likely `to die` in this process and being discarded.

6. Estimation error

To evaulate the accuracy of the Particle Filter its **positioning accuracy** is measured given the ground truth position of the car. For the purpose of this project the accuracy is evaluated computing a weighted average error of all particles used by the filter.

# Running the Code

This project involves the Term 2 Simulator which can be downloaded [here](https://github.com/udacity/self-driving-car-sim/releases)

This repository includes two files that can be used to set up and intall uWebSocketIO for either Linux or Mac systems. For windows you can use either Docker, VMware, or even Windows 10 Bash on Ubuntu to install uWebSocketIO.

Once the install for uWebSocketIO is complete, the main program can be built and ran by doing the following from the project top directory.

`./build.sh` 

`./run.sh` 

At this point run the Term 2 Simulator, select "Project 3: Kidnapped Vehicle" and start the simulation hitting the 'Start' button. Once the simulation starts you should see a car moving around, together with a circle representing the estimated car position and an arrow representing the estimated car heading provided by the Particle Filter. Landmarks associated to measurements are also identified during the simulation.

# Project structure and code

The source code provided in `main.cpp` implements communication with the simulator.

Here is the main protcol used by `main.cpp` to communicate with the simulator.

INPUT: values provided by the simulator to the C++ program

// sense noisy position data from the simulator

["sense_x"] 

["sense_y"] 

["sense_theta"] 

// get the previous velocity and yaw rate to predict the particle's transitioned state

["previous_velocity"]

["previous_yawrate"]

// receive noisy observation data from the simulator, in a respective list of x/y values

["sense_observations_x"] 

["sense_observations_y"] 


OUTPUT: values provided by the c++ program to the simulator

// best particle values used for calculating the error evaluation

["best_particle_x"]

["best_particle_y"]

["best_particle_theta"] 

// Optional message data used for debugging particle's sensing and associations

// for respective (x,y) sensed positions ID label 

["best_particle_associations"]

// for respective (x,y) sensed positions

["best_particle_sense_x"] <= list of sensed x positions

["best_particle_sense_y"] <= list of sensed y positions


The directory structure of this repository is as follows:

```
root
|   build.sh
|   clean.sh
|   CMakeLists.txt
|   README.md
|   run.sh
|
|___data
|   |   
|   |   map_data.txt
|   
|   
|___src
    |   helper_functions.h
    |   main.cpp
    |   map.h
    |   particle_filter.cpp
    |   particle_filter.h
```

## Inputs to the Particle Filter
You can find the inputs to the particle filter in the `data` directory. 

#### The Map*
`map_data.txt` includes the position of landmarks (in meters) on an arbitrary Cartesian coordinate system. Each row has three columns
1. x position
2. y position
3. landmark id

### All other data the simulator provides, such as observations and controls.

> * Map data provided by 3D Mapping Solutions GmbH.