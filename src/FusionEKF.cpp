#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>
#include <cmath>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

namespace {
  const float PI = 3.14159;
  const int noise_ax = 9;
  const int noise_ay = 9;

  float computeProcessCovariance(const float delt, KalmanFilter& ekf)
  {
    const float t2 = pow(delt, 2);
    const float t3 = pow(delt,3)/2;
    const float t4 = pow(delt,4)/4;

    const float ax2 = t2 * noise_ax;
    const float ay2 = t2 * noise_ay;
    const float ax3 = t3 * noise_ax;
    const float ay3 = t3 * noise_ay;
    const float ax4 = t4 * noise_ax;
    const float ay4 = t4 * noise_ay;

    ekf.Q_ <<
      ax4, 0, ax3, 0,
      0, ay4, 0, ay3,
      ax3, 0, ax2, 0,
      ay3, 0, ay2, 0;
  }
}

/*
 * Constructor.
 */
FusionEKF::FusionEKF()
  :is_initialized_(false), previous_timestamp_(0),
   R_laser_(MatrixXd(2,2)),  R_radar_(MatrixXd(3, 3)),
   H_laser_(MatrixXd(2, 4)), Hj_(MatrixXd(3, 4))
{
  //measurement covariance matrix - laser
  R_laser_ <<
    0.0225, 0,
    0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ <<
    0.09, 0, 0,
    0, 0.0009, 0,
    0, 0, 0.09;

  // Laser prediction-measurement matrix (px and py)
  H_laser_ <<
    1, 0, 0, 0,
    0, 1, 0, 0;

  std::cout << "FusionEKF construction complete" << std::endl;
}

void
FusionEKF::initFilter(const MeasurementPackage& measurement)
{
  VectorXd initialState(4);
  if (measurement.sensor_type_ == MeasurementPackage::LASER) {
    std::cout << "Initializing using laser measurement" << std::endl;
    // record the initial x and y position
    initialState << measurement.raw_measurements_[0], measurement.raw_measurements_[1];
    std::cout << "State initialization complete" << std::endl;
  } else { // if (sensorType == MeasurementPackage::RADAR
    std::cout << "Initializing using radar measurement" << std::endl;
    // derive the initial x and y positions from the polar co-ordinates
    const float rho = measurement.raw_measurements_[0];
    const float phi = measurement.raw_measurements_[1];
    const float px = rho * cos (phi * PI/180);
    const float py = rho * sin (phi * PI/180);
    initialState << px, py;
    std::cout << "State initialization complete" << std::endl;

    // Radar Jacobian matrix (linear approximation of polar to cartesian)
    Hj_ << tools_.CalculateJacobian(ekf_.x_);
    std::cout << "Hj initialization" << std::endl;
  }

   // initialize the state transition matrix
  // this is the first measurement and so delt is 0
  MatrixXd stateTransition(4,4);
  stateTransition << 1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1;
  std::cout << "State transition initialization complete" << std::endl;

  // initialize the state co-variance matrix
  MatrixXd stateCovariance(4,4);
  // we are certain about the position but uncertain about the velocity
  stateCovariance << 1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1000, 0,
    0, 0, 0, 1000;
  std::cout << "State co-variance initialization complete" << std::endl;

  MatrixXd processCovariance(4,4);
  // since delt is 0, this will all be 0s (no acceleration noise)
  processCovariance << 0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0;
  std::cout << "Process co-variance initialization complete" << std::endl;

  ekf_.Init(initialState, stateCovariance, stateTransition, processCovariance);
  previous_timestamp_ = measurement.timestamp_;
  std::cout << "Filter initialization complete" << std::endl;
  is_initialized_ = true;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    initFilter(measurement_pack);
  }
    
  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  // delt in secs
  const float delt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  
  // update the state transition matrix to include the delt
  ekf_.F_(0,2) = delt;
  ekf_.F_(1,3) = delt;
  // update the process co-variance
  computeProcessCovariance(delt, ekf_);

  // perform the prediction
  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
    // use regular Kalman Filter equations
    ekf_.Update(measurement_pack.raw_measurements_, H_laser_, R_laser_);
  } else {
    // use Extended Kalman Filter equations
    // update Jacobian
    Hj_ << tools_.CalculateJacobian(ekf_.x_);
    ekf_.UpdateEKF(measurement_pack.raw_measurements_, Hj_, R_radar_);
  }
  
  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
