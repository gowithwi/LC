/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

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

void ParticleFilter::init(double gps_x, double gps_y, double gps_theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    num_particles = 1000;
    default_random_engine gen;

    double std_x = std[0];
    double std_y = std[1];
    double std_theta = std[2];

    for (int i = 0; i < num_particles; ++i) {

        normal_distribution<double> dist_x(gps_x, std_x);
        normal_distribution<double> dist_y(gps_y, std_y);
        normal_distribution<double> dist_theta(gps_theta, std_theta);

        double sample_x, sample_y, sample_theta;
        
        sample_x = dist_x(gen);
        sample_y = dist_y(gen);
        sample_theta = dist_theta(gen);
        particles[i].x=sample_x;
        particles[i].y=sample_y;
        particles[i].theta=sample_theta;
        particles[i].weight=1.0;
        weights[i] = particles[i].weight;
    }
    is_initialized = true;
    
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    for (int i = 0; i < num_particles; ++i) {

        if (yaw_rate == 0){
            particles[i].x += velocity*delta_t*cos(particles[i].theta);
            particles[i].y += velocity*delta_t*sin(particles[i].theta);
        }else{
            particles[i].x += velocity/yaw_rate*(sin(particles[i].theta+yaw_rate*delta_t)-sin(particles[i].theta));
            particles[i].y += velocity/yaw_rate*(cos(particles[i].theta)-cos(particles[i].theta+yaw_rate*delta_t));
            particles[i].theta += yaw_rate*delta_t;
        }
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    
    for(unsigned int i = 0; i < observations.size(); i++){

        double distance_min = dist(predicted[0].x,predicted[0].y,observations[i].x,observations[i].y);
        observations[i].id = 0;

        for(unsigned int j = 1; j < predicted.size(); j++){
            
            double distance = dist(predicted[j].x,predicted[j].y,observations[i].x,observations[i].y);
            if (distance<distance_min){
                distance_min = distance;
                observations[i].id = j;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

    // 1. I need to convert the data from map class to LandmarkObs structure.
    std::vector<LandmarkObs> XY_M;
    for(unsigned int k = 0; k < map_landmarks.landmark_list.size(); k++){ // for each observation, convert it from car coordinates to map coordinates
        XY_M[k].id = map_landmarks.landmark_list[k].id_i;
        XY_M[k].x = map_landmarks.landmark_list[k].x_f;
        XY_M[k].y = map_landmarks.landmark_list[k].y_f;
    }
    
    // 2. Now I need to play with each particle,
    for(unsigned int i = 0; i < num_particles; i++){ // for each particle, run all observations
        
        std::vector<LandmarkObs> XY_O;
        
        for(unsigned int j = 0; j < observations.size(); j++){ // for each observation, convert it from car coordinates to map coordinates
    // 2.(a) convert the observation from car coord. to map coord.
            std::vector<double> Observed_In_Map = Convert_Car2Map(observations[j].x, observations[j].x, particles[i].x, particles[i].y, particles[i].theta);
            
            XY_O[j].id = 0; // let's change this later in the association.
            XY_O[j].x = Observed_In_Map[0];
            XY_O[j].y = Observed_In_Map[1];
        }
    // 2.(b) associate the observsation (in map coordinates) to the landmarks.
            dataAssociation(XY_M, XY_O);
        
    // 2.(c) determine the total weight for each particle
        for(unsigned int j = 0; j < XY_O.size(); j++){
            
            double par_x = (XY_O[j].x-XY_M[j].x)/std_landmark[0];
            double par_y =(XY_O[j].y-XY_M[j].y)/std_landmark[1];
            double exp_index = - 0.5*(par_x*par_x + par_y*par_y);
            double local_weight = 1.0/(2*M_PI*std_landmark[0]*std_landmark[1])*exp(exp_index);
            particles[i].weight *= local_weight;
            weights[i] = particles[i].weight;
        }
    }
    
    // I did not use the sensor range here; if the estimation is out of range, it is a bad particle, and deserves a tinty weight.
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    
    // 1. generate the random distribution.
    default_random_engine generator;
    discrete_distribution<double> distribution (weights.begin(),weights.end());
    
    vector<int> numnum;
    for (unsigned int i=0; i<num_particles; ++i) {
        numnum[i] = distribution(generator);
    }

    // 2. I need to generate a mirror particle arrays to store the current particle arrays
    vector<Particle> particles_mirror;
    particles_mirror = particles;
    
    // 3. Now I write in the resampled particles.
    for (unsigned int i=0; i<num_particles; ++i) {
        int index = numnum[i];
        particles[i] = particles_mirror[index];
        particles[i].weight = 1.0;
    }

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
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
