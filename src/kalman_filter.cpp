#include "kalman_filter.h"
#include "tools.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  Q_ = Q_in;

  long x_size = x_.size();
  I_ = MatrixXd::Identity(x_size, x_size);
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  P_ = (F_ * P_) * F_.transpose() + Q_;
}

void
KalmanFilter::Update(const Eigen::VectorXd &z,
		     const Eigen::MatrixXd& H,
		     const Eigen::MatrixXd& R)
{
  VectorXd y = z - H * x_;
  const MatrixXd Ht = H.transpose();
  MatrixXd S = H * P_ * Ht + R;
  MatrixXd K = P_ * Ht * S.inverse();

  // new estimates
  x_ = x_ + (K * y);
  P_ = (I_ - K * H) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z,
			     const MatrixXd& Hj,
			     const MatrixXd& R) 
{
  // convert from cartesian predictions to polar measurements
  double px = x_[0];
  double py = x_[1];
  const double vx = x_[2];
  const double vy = x_[3];
  const double rho = sqrt(pow(px,2) + pow(py, 2));

  if (std::abs(py) < Tools::EPS_ && std::abs(px) < Tools::EPS_) {
    px = Tools::EPS_;
    py = Tools::EPS_;
  }
  const double phi = atan2(py, px);
  double rhoRate = 0;
  if (std::abs(rho) > Tools::EPS_) {
    rhoRate = (px * vx + py * vy) / rho;
  }

  VectorXd polarState(3);
  polarState << rho, phi, rhoRate;
  
  VectorXd y = z - polarState;
  // normalize angle after subtraction
  y[1] = atan2(sin(y[1]), cos(y[1]));

  const MatrixXd Ht = Hj.transpose();
  MatrixXd S = Hj * P_ * Ht + R;
  MatrixXd K = P_ * Ht * S.inverse();

  // new estimates
  x_ = x_ + (K * y);
  P_ = (I_ - K * Hj) * P_;
}
