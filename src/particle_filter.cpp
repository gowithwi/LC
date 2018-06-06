/*
 * 0605-4
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

    num_particles = 100;
//    default_random_engine gen;
    
    normal_distribution<double> dist_x(gps_x, std[0]);
    normal_distribution<double> dist_y(gps_y, std[1]);
    normal_distribution<double> dist_theta(gps_theta, std[2]);

    for (int i = 0; i < num_particles; ++i) {

        Particle particle;
        particle.id = i;
        particle.x=dist_x(gen);
        particle.y=dist_y(gen);
        particle.theta=dist_theta(gen);
        particle.weight=1.0;
        
        particles.push_back(particle);
        weights.push_back(1.0);
        cout<<particle.x<<"/n";
    }
    is_initialized = true;
    
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    
//    default_random_engine gen;
    double new_x;
    double new_y;
    double new_theta;
    
    for (int i = 0; i < num_particles; ++i) {

        if ( abs(yaw_rate) > 0.0){
            new_x = particles[i].x + velocity/yaw_rate*(sin(particles[i].theta+yaw_rate*delta_t)-sin(particles[i].theta));
            new_y = particles[i].y + velocity/yaw_rate*(cos(particles[i].theta)-cos(particles[i].theta+yaw_rate*delta_t));
            new_theta = particles[i].theta + yaw_rate*delta_t;

        }else{
            new_x = particles[i].x + velocity*delta_t*cos(particles[i].theta);
            new_y = particles[i].y + velocity*delta_t*sin(particles[i].theta);
            new_theta = particles[i].theta;
        }
        normal_distribution<double> dist_x(new_x, std_pos[0]);
        normal_distribution<double> dist_y(new_y, std_pos[1]);
        normal_distribution<double> dist_theta(new_theta, std_pos[2]);
        particles[i].x = dist_x(gen);
        particles[i].y = dist_y(gen);
        particles[i].theta = dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // I don't need this one
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

    for (int p=0;p<num_particles;p++){  // run through all particles
        
        vector<int> associations;
        vector<double> sense_x;
        vector<double> sense_y;
        
        //  vector<LandmarkObs> observations;
        vector<LandmarkObs> trans_observations;
        LandmarkObs trans_obs;
        
        for(unsigned int i=0;i<observations.size();i++){ // convert all observations into global coordinates
            
            double xc = observations[i].x;
            double yc = observations[i].y;
            double xp = particles[p].x;
            double yp = particles[p].y;
            double yaw = particles[p].theta;
            
            trans_obs.x = cos(yaw)*xc - sin(yaw)*yc + xp;
            trans_obs.y = sin(yaw)*xc + cos(yaw)*yc + yp;
            trans_obs.id = i;
            trans_observations.push_back(trans_obs);
        }
        
        particles[p].weight = 1; // re-initialize the weight
        
        for(unsigned int i=0;i<trans_observations.size();i++) // match each observation...
        {
            double closest_dis = sensor_range; // set default
            int association = -1;
            for(unsigned int j=0;j<map_landmarks.landmark_list.size();j++) // ...to a single point in the map
                {
                    double landmark_x = map_landmarks.landmark_list[j].x_f;
                    double landmark_y = map_landmarks.landmark_list[j].y_f;
                    
                    double dis = dist(landmark_x, landmark_y, trans_observations[i].x, trans_observations[i].y);
                    if(dis<closest_dis)
                    {
                        closest_dis = dis;
                        association = j; // remember that!
                    }
                }
            

            if(association>-1){
                double meas_x = trans_observations[i].x;
                double meas_y = trans_observations[i].y;
                double mu_x = map_landmarks.landmark_list[association].x_f;
                double mu_y = map_landmarks.landmark_list[association].y_f;
                long double multiplier = 1/(2*M_PI*std_landmark[0]*std_landmark[1]);
                long double exp_ind = pow((meas_x-mu_x)/std_landmark[0],2) + pow((meas_y-mu_y)/std_landmark[1],2);
                multiplier *= exp(-0.5*exp_ind);
                particles[p].weight *= multiplier;
            }else {
                
                long double penalty = 1/(2*M_PI*std_landmark[0]*std_landmark[1]);
                long double exp_ind = pow(sensor_range/std_landmark[0],2) + pow(sensor_range/std_landmark[1],2);
                penalty *= exp(-0.5*exp_ind);
                particles[p].weight *= penalty;
            }
            
            associations.push_back(association); // push in each observations
            sense_x.push_back(trans_observations[i].x);
            sense_y.push_back(trans_observations[i].y);
        }
        
        particles[p] = SetAssociations(particles[p],associations,sense_x,sense_y); //
        weights[p] =particles[p].weight;
        
        associations.clear();
        sense_x.clear();
        sense_y.clear();
        trans_observations.clear();
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    
//    default_random_engine gen;
    discrete_distribution<int> distribution(weights.begin(),weights.end());
    
    vector<Particle> resample_particles;
    
    for ( int i=0; i<num_particles; ++i) {
        resample_particles.push_back(particles[distribution(gen)]);
    }
    particles = resample_particles;
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
