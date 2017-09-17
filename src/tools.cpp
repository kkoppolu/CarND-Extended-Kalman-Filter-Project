#include <iostream>
#include <cmath>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

const double Tools::EPS_ = 1e-4;
const double Tools::PI_ = 3.14159;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth)
{
  std::cout << "Estimating RMSE" << std::endl;
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  if (estimations.size() == 0 || estimations.size() != ground_truth.size()) {
    std::cerr << "CalculateRMSE () - Error - Malformed inputs" << endl;
    return rmse;
  }
  
  for (int i = 0; i < estimations.size(); ++i) {
    VectorXd residualSum = estimations[i] - ground_truth[i];
    residualSum = residualSum.array() * residualSum.array();
    rmse += residualSum;
  }

  rmse /= estimations.size();
  
  rmse = rmse.array().sqrt();

  std::cout << "RMSE: " << rmse << std::endl;
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd Hj(3,4);
  //recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  //pre-compute a set of terms to avoid repeated calculation
  float c1 = px*px + py*py;
  float c2 = sqrt(c1);
  float c3 = (c1*c2);

  //check division by zero
  if(std::abs(c1) < EPS_ || std::abs(c2) < EPS_){
    std::cerr << "CalculateJacobian () - Error - Division by Zero" << endl;
    return Hj;
  }

  //compute the Jacobian matrix
  Hj << (px/c2), (py/c2), 0, 0,
    -(py/c1), (px/c1), 0, 0,
    py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

  return Hj;
}
