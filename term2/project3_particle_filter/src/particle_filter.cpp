#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

  // number of particles (must be tuned to balance estimation accuracy and filter speed)
  num_particles = 30;

  // Resize the vector of weights to match the number of particles in the
  // Particle filter
  weights.resize(num_particles);

  // Resize the vector of particles to match the number of particles in the
  // Particle Filter
  particles.resize(num_particles);

  // Random number generator for initial particle position
  std::default_random_engine gen;

  // Normal distribution generator for x coordinate
  std::normal_distribution<double> dist_x(x, std[0]);

  // Normal distribution generator for y coordinate
  std::normal_distribution<double> dist_y(y, std[1]);

  // Normal distribution generator for theta (heading)
  std::normal_distribution<double> dist_theta(theta, std[2]);

  // Iterate over all particles initializing each
  // using coordinates provided by the GPS
  for (int i = 0; i < num_particles; ++i)
  {
  	particles[i].id = i; // id
  	particles[i].x = dist_x(gen); // initial x position
  	particles[i].y = dist_y(gen); // initial y position
  	particles[i].theta = dist_theta(gen); // initial theta (heading)
  	particles[i].weight = 1.0; // initial weight
  }

  // Particle Filter initialized
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	
	// Random number generator
  std::default_random_engine gen;

  // Normal distribution generator for x coordinate
  std::normal_distribution<double> dist_x(0.0, std_pos[0]);

  // Normal distribution generator for y coordinate
  std::normal_distribution<double> dist_y(0.0, std_pos[1]);

  // Normal distribution generator for theta (heading)
  std::normal_distribution<double> dist_theta(0.0, std_pos[2]);

	// Predict the position of ech particle using the motion model
	for (int i = 0; i < num_particles; ++i)
	{
		// yaw_rate == 0 (going straight)
		if (fabs(yaw_rate) < 0.001)
		{
			particles[i].x += velocity * delta_t + cos(particles[i].theta);
			particles[i].y += velocity * delta_t + sin(particles[i].theta);
			//particles[i].theta = particles[i].theta;
		}
		else // yaw rate != 0
		{
			particles[i].x += velocity/yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			particles[i].y += velocity/yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}

		// Add random Gaussian noise to each particle to account
		// for uncertainty in control input 
		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the
	// observed measurement to this particular landmark.

	// Iterate over all observed measurements
	for (int i = 0; i < observations.size(); ++i)
	{
		int smallest_j = 0;
		double min_distance = std::numeric_limits<double>::max();

		// Iterate over all predicted measurements
		// finding the closest observed measurement
		for (int j = 0; j < predicted.size(); ++j)
		{
			double dx = predicted[j].x - observations[i].x;
			double dy = predicted[j].y - observations[i].y;
			double distance = dx * dx + dy * dy;
			if (distance < min_distance)
			{
				min_distance = distance;
				smallest_j = j;
			}
		}

		// Predicted measurement at index 'smallest_j' is the closest
		// predicted measurement for observed measurement 'i'
		observations[i].id = smallest_j;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution.
	// Measurement observations are given in the vehicle's coordinate system.
	// Particles are located according to the map's coordinate system.
	// As a result transformation between vehicle reference system and map reference
	// system is required.
	// This transformation requires both rotation AND translation (but no scaling).
	// Values used to compute Multivariate-Gaussian
	// (declared outsize of for loop since values
	// do not depend on i/j)
	
	// Landmark position standard deviations
	double stdx = std_landmark[0];
	double stdy = std_landmark[1];

	// Helper coefficients for Multivariate Gaussian distribution
	double coeffx = 1.0 / (2.0 * stdx * stdx);
	double coeffy = 1.0 / (2.0 * stdy * stdy);
	double coeff = 1.0 / (2.0 * M_PI * stdx * stdy);

	// Iterate over all particles to update each particle's weight
	for (int i = 0; i < num_particles; ++i)
	{
		double px = particles[i].x;
		double py = particles[i].y;
		double theta = particles[i].theta;

		// List of observations in map coordinates
		std::vector<LandmarkObs> map_observations;

		// Measurement observations are transformed from vehicle
		// reference system to map reference system)
		for (int j = 0; j < observations.size(); ++j)
		{
			// Obaservation ID, x and y position
			int tid = observations[j].id;
			double tx = px + observations[j].x * cos(theta) - observations[j].y * sin(theta);
			double ty = py + observations[j].y * cos(theta) + observations[j].x * sin(theta);

			// Add the transformed observation to the list of
			// observations in map coordinates
			LandmarkObs obs = {tid, tx, ty};
			map_observations.push_back(obs);
		}

		// List of landmarks in range of sensor
		std::vector<LandmarkObs> landmarks_in_range;

		// Find landmarks which are in sensor range
		for (int j = 0;  j < map_landmarks.landmark_list.size(); ++j)
		{
			// Landmark id and position
			int landmark_id = map_landmarks.landmark_list[j].id_i;
			double landmark_x = map_landmarks.landmark_list[j].x_f;
			double landmark_y = map_landmarks.landmark_list[j].y_f;

			double dx = landmark_x - px;
			double dy = landmark_y - py;

			// Compute distance between the landmark and the sensor
			double distance = sqrt(dx * dx + dy * dy);

			// If the landmark is in sensor range add it to the
			// list of landmarks in range
			if (distance < sensor_range)
			{
				LandmarkObs landmark = {landmark_id, landmark_x, landmark_y};
				landmarks_in_range.push_back(landmark);
			}
		}

		// Associate landmarks in range of sensor to observations on the map
		dataAssociation(landmarks_in_range, map_observations);

		// Weight computation
		// Use Multivariate-Gaussian distribution to derive the
		// weight of each particle.
		// P(x,y): x and y are measurement observations in map coordinates
		// mu_x, mu_y: landmark position
		// stdx, stdy: standard deviation on landmark position 
		double w = 1.0;

		for (int j = 0; j < map_observations.size(); ++j)
		{
			int mid = map_observations[j].id;
			double mx = map_observations[j].x;
			double my = map_observations[j].y;

			double px = landmarks_in_range[mid].x;
			double py = landmarks_in_range[mid].y;

			double dx = mx - px;
			double dy = my - py;
			double gaussian = coeff * exp(-(coeffx * dx * dx + coeffy * dy * dy));

			// Each particle weight is the product of all weights
			// obtained considering all measurement observations
			w = w * gaussian;
		}

		// Update the particle weight
		particles[i].weight = w;

		// Add the weight to the list of weights
		weights[i] = w;
	}
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to each particle's weight.
	
	// New set of particles created doing resampling
	// This set of particles will replace the current
	// particles in the next iteration (next time step)
	std::vector<Particle> new_particles;

	// Random number generator
	std::default_random_engine gen;

	// Discrtete distribution used to select random weight indexes
	std::discrete_distribution<int> w_index(weights.begin(), weights.end());

	for (int i = 0; i < num_particles; ++i)
	{
		// Generate a random weight index
		int index = w_index(gen);

		// Create a new particle out of the randomly selected particle
		Particle p = {
			index, // id
			particles[index].x, // x position
			particles[index].y, // y position
			particles[index].theta, // theta (heading)
			particles[index].weight // weight
		};

		// Add the particle to the set of particles
		// to use in the next iteration
		new_particles.push_back(p);
	}

	// Replace current set of particles with the
	// new resampled set of particles
	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	// particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	// Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
	copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length()-1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
	copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length()-1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
	copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length()-1);  // get rid of the trailing space
	return s;
}
