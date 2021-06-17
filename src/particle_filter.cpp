/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

static std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 200;  // TODO: Set the number of particles
  
  std::default_random_engine gen; // Creating the gaussian engine
  
  // Unpacking the standard deviations
  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2]; 
  
  //Setting the gaussians
  std::normal_distribution<double> dist_x(x,std_x);
  std::normal_distribution<double> dist_y(y,std_y);
  std::normal_distribution<double> dist_theta(theta,std_theta);
  
  //Creating the particles and adding them to vector<Particles> particles
  for (unsigned int i=0;i<num_particles;i++){
    Particle sample;
    sample.id = i;
    sample.x = dist_x(gen);
    sample.y = dist_y(gen);
    sample.theta = dist_theta(gen);
    sample.weight = 1;
    particles.push_back(sample);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  
  //Setting the gaussians noises functions
  std::normal_distribution<double> noise_x(0, std_pos[0]);
  std::normal_distribution<double> noise_y(0, std_pos[1]);
  std::normal_distribution<double> noise_theta(0, std_pos[2]);
 

  for (int i=0;i<num_particles;i++){
    
    //If yaw_rate is too small, use a constant yaw model
    float min_yaw_rate = 0.0001;
    if (fabs(yaw_rate) < min_yaw_rate) {  
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    } 
    else {
      particles[i].x += velocity/yaw_rate*(sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y += velocity/yaw_rate*(cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }
    
    //Add the noise
    particles[i].x += noise_x(gen);
    particles[i].y += noise_y(gen);
    particles[i].theta += noise_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

  for (unsigned i=0;i<observations.size();i++){
    LandmarkObs best_predict;
    double distance, distance_min;
    LandmarkObs current_obs = observations[i];
    int index = 0;
    for (unsigned j=0;j<predicted.size();j++){
      LandmarkObs current_predict = predicted[j];
  
      distance = dist(current_predict.x,current_predict.y,current_obs.x,current_obs.y);
      
      if (j==0){
        best_predict = current_predict;
        distance_min = distance;
      } 
      else{
        if (distance < distance_min){
          best_predict = current_predict;
          distance_min = distance;
          index = j;
        }
      }
 
    }     
    observations[i] = best_predict;
  }  
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  //For each particle in the filter:
  //const double lowest_double = -std::numeric_limits<double>::max();
  for (unsigned i=0;i<num_particles;i++){
    Particle sample = particles[i];
    double p_x = sample.x;
	double p_y = sample.y;
	double theta = sample.theta;
  
    vector<LandmarkObs> observations_map, associated_map;
       
    //First, convert each observations into the global/map coordinates:
    for (unsigned j=0;j<observations.size();j++){
      LandmarkObs obs=observations[j];
      LandmarkObs obs_map;
      obs_map.x = p_x + cos(theta)*obs.x - sin(theta)* obs.y;
      obs_map.y = p_y + sin(theta)*obs.x + cos(theta)* obs.y;
      observations_map.push_back(obs_map); 
    }
    
    // Now observations_map is a vector of the observations in Map coordinate
        
    //Create <LandmarkObs> vector of the landmarks in map 
    vector<LandmarkObs> map_landmarks_cropped;
    for (unsigned mark=0;mark<map_landmarks.landmark_list.size();mark++){
      LandmarkObs point;
      point.x = map_landmarks.landmark_list[mark].x_f;
      point.y = map_landmarks.landmark_list[mark].y_f;
      //double distance = dist(point.x,point.y, p_x ,p_y);
      
      //Using a square for searching the landmarks (faster than a circle)
      if ((fabs(p_x-point.x) <=sensor_range )&&( fabs(p_y-point.y)<=sensor_range)){
        map_landmarks_cropped.push_back(point); //Cropping the list of landmarks into the sense_range 
      }
    }
    
    // Now map_landmarks_cropped is a vector of map landmarks within the sensor_range value centered in the particle
    
    //Data Association: Associate each observation with the best map landmark    
    associated_map = observations_map;
    
    dataAssociation(map_landmarks_cropped, associated_map); //Edit associated_map to the associated points
      
    //Now associated_map is a vector of the associated map landmarks.
    
    //Rinit weight
    particles[i].weight = 1;
    double weight_temp = 1;
    
    //Compute the bigaussian for each pair
    for (unsigned obs=0;obs<associated_map.size();obs++){

      //Bigaussian computation:
      LandmarkObs p1, p2;
      p1 = associated_map[obs];
      p2 = observations_map[obs];
      double sigma_x = std_landmark[0];
      double sigma_y = std_landmark[1];     
      double coef =1/(2*M_PI*sigma_x*sigma_y);
      double coef_x=pow((p1.x-p2.x),2) / (2*pow(sigma_x,2));
      double coef_y=pow((p1.y-p2.y),2) / (2*pow(sigma_y,2));
      
      //Finally, update each particle weight
      weight_temp = coef*exp(-coef_x-coef_y);
      particles[i].weight*=weight_temp;
  
    }   
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
  //Initializing the weights vector
  vector<double> weights;
  for (unsigned i=0;i<num_particles;i++){
    weights.push_back(particles[i].weight);
  }
  
  //weights is a vector of all the weights
  
  //Setting the generators
  std::default_random_engine generator;
  std::discrete_distribution<int> distribution(weights.begin(), weights.end());
  
  //Resampling
  vector<Particle> particles_temp;
  for (unsigned i=0;i<num_particles;i++){
    int index = distribution(generator);
    particles_temp.push_back(particles[index]);
  }
  
  //Updating the particles
  particles = particles_temp;
  
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}