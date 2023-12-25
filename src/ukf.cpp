#include "ukf.h"
#include "Eigen/Dense"
#include "measurement_package.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  is_initialized_ = false;
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  // initialized with first measurement
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd::Identity(5, 5) * 0.5;

  Xsig_pred_ = MatrixXd(5, 15);
  Xsig_pred_.fill(0.0);

  // augmented sigma points
  Xsig_aug_ = MatrixXd(7, 15);
  Xsig_aug_.fill(0.0);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3.0;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1.0;

  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
   * End DO NOT MODIFY section for measurement noise values
   */
  n_x_ = 5;
  n_aug_ = 7;
  n_z_ = 3;
  lambda_ = 3.0 - n_x_;
  weights_ = VectorXd(2 * n_aug_ + 1);
  double weight_0 = lambda_ / (lambda_ + n_aug_);
  double weight = 0.5 / (lambda_ + n_aug_);
  weights_(0) = weight_0;

  for (int i = 1; i < 2 * n_aug_ + 1; ++i) {
    weights_(i) = weight;
  }

  // create matrix for sigma points in measurement space
  Zsig_ = MatrixXd(n_z_, 2 * n_aug_ + 1);
  Zsig_.fill(0.0);

  // laser measurement matrix
  H_ = MatrixXd(2, n_x_);
  H_ << 1, 0, 0, 0, 0, 0, 1, 0, 0, 0;

  // laser measurement covariance matrix
  RL_ = MatrixXd(2, 2);
  RL_ << std_laspx_ * std_laspx_, 0, 0, std_laspy_ * std_laspy_;

  // radar measurement covariance matrix
  RR_ = MatrixXd(n_z_, n_z_);
  RR_ << std_radr_ * std_radr_, 0, 0, 0, std_radphi_ * std_radphi_, 0, 0, 0,
      std_radrd_ * std_radrd_;
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(const MeasurementPackage &meas_package) {
  if (!is_initialized_) {
    switch (meas_package.sensor_type_) {
    case MeasurementPackage::LASER:
      x_.head(2) = meas_package.raw_measurements_;
      x_.tail(3).fill(0.0);
      time_ = meas_package.timestamp_;
      is_initialized_ = true;
    default:
      return;
    }
  }

  // debug output
  // std::cout << "x_" << std::endl;
  // std::cout << x_ << std::endl;
  // std::cout << "P_" << std::endl;
  // std::cout << P_ << std::endl;
  // std::cout << "Xsig_pred_" << std::endl;
  // std::cout << Xsig_pred_ << std::endl;
  // std::cout << "Xsig_aug_" << std::endl;
  // std::cout << Xsig_aug_ << std::endl;

  auto delta_t = meas_package.timestamp_ - time_;
  // update the timestamp
  time_ = meas_package.timestamp_;
  Prediction(delta_t / 1.0e6);

  switch (meas_package.sensor_type_) {
  case MeasurementPackage::LASER:
    // std::cout << "laser measurement" << std::endl;
    if (use_laser_) UpdateLidar(meas_package);
    break;
  case MeasurementPackage::RADAR:
    // std::cout << "radar measurement" << std::endl;
    if (use_radar_) UpdateRadar(meas_package);
    break;
  default:
    std::cout << "Error: Unknown measurement source.";
    exit(1);
  }
}

void UKF::Prediction(const double delta_t) {
  // compute augmented sigma points
  AugmentedSigmaPoints();
  // predict next sigma points
  SigmaPointPrediction(delta_t);
  // predict mean and covariance from sigma points
  PredictMeanAndCovariance();
}

void UKF::UpdateLidar(const MeasurementPackage &meas_package) {
  VectorXd z = meas_package.raw_measurements_.head(2);
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + RL_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  // new estimate
  x_ = x_ + (K * y);
  MatrixXd I = MatrixXd::Identity(5, 5);
  P_ = (I - K * H_) * P_;
  // std::cout << "done update lidar" << std::endl;
}

void UKF::UpdateRadar(const MeasurementPackage &meas_package) {
  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z_);
  MatrixXd S = MatrixXd(n_z_, n_z_);
  PredictRadarMeasurement(z_pred, S);
  VectorXd z = meas_package.raw_measurements_;

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z_);

  // calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) { // 2n+1 simga points
    // residual
    VectorXd z_diff = Zsig_.col(i) - z_pred;
    // angle normalization
    while (z_diff(1) > M_PI)
      z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI)
      z_diff(1) += 2. * M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    while (x_diff(3) > M_PI)
      x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI)
      x_diff(3) += 2. * M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // residual
  VectorXd z_diff = z - z_pred;

  // angle normalization
  while (z_diff(1) > M_PI)
    z_diff(1) -= 2. * M_PI;
  while (z_diff(1) < -M_PI)
    z_diff(1) += 2. * M_PI;

  // update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  // std::cout << "done update radar" << std::endl;
}

void UKF::AugmentedSigmaPoints() {

  // create augmented mean vector
  VectorXd x_aug = VectorXd(7);

  // create augmented state covariance
  MatrixXd P_aug = MatrixXd(7, 7);

  // create augmented mean state
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  // create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5, 5) = P_;
  P_aug(5, 5) = std_a_ * std_a_;
  P_aug(6, 6) = std_yawdd_ * std_yawdd_;

  // create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  // create augmented sigma points
  Xsig_aug_.col(0) = x_aug;
  for (int i = 0; i < n_aug_; ++i) {
    Xsig_aug_.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug_.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }
  // std::cout << "done augment sigma points" << std::endl;
}

void UKF::SigmaPointPrediction(const double delta_t) {
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    // extract values for better readability
    double p_x = Xsig_aug_(0, i);
    double p_y = Xsig_aug_(1, i);
    double v = Xsig_aug_(2, i);
    double yaw = Xsig_aug_(3, i);
    double yawd = Xsig_aug_(4, i);
    double nu_a = Xsig_aug_(5, i);
    double nu_yawdd = Xsig_aug_(6, i);

    // predicted state values
    double px_p, py_p;

    // avoid division by zero
    if (fabs(yawd) > 0.001) {
      px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
    } else {
      px_p = p_x + v * delta_t * cos(yaw);
      py_p = p_y + v * delta_t * sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    // add noise
    px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p = v_p + nu_a * delta_t;

    yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p = yawd_p + nu_yawdd * delta_t;

    // write predicted sigma point into right column
    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }
  // std::cout << "done sigma point prediction" << std::endl;
}

void UKF::PredictMeanAndCovariance() {
  // predicted state mean
  x_.fill(0.0);
  // iterate over sigma points
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }

  // predicted state covariance matrix
  P_.fill(0.0);
  // iterate over sigma points
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    while (x_diff(3) > M_PI) {
      x_diff(3) -= 2. * M_PI;
    }
    while (x_diff(3) < -M_PI)
      x_diff(3) += 2. * M_PI;

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }
  // std::cout << "done predict mean/covariance" << std::endl;
}

void UKF::PredictRadarMeasurement(VectorXd &z_pred, MatrixXd &S) {
  // transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) { // 2n+1 simga points
    // extract values for better readability
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;

    // measurement model
    Zsig_(0, i) = sqrt(p_x * p_x + p_y * p_y);                         // r
    Zsig_(1, i) = atan2(p_y, p_x);                                     // phi
    Zsig_(2, i) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y); // r_dot
  }

  // mean predicted measurement
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    z_pred = z_pred + weights_(i) * Zsig_.col(i);
  }

  // innovation covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) { // 2n+1 simga points
    // residual
    VectorXd z_diff = Zsig_.col(i) - z_pred;

    // angle normalization
    while (z_diff(1) > M_PI)
      z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI)
      z_diff(1) += 2. * M_PI;

    S += weights_(i) * z_diff * z_diff.transpose();
  }

  // add measurement noise covariance matrix
  S += RR_;
  // std::cout << "done predict radar measurement" << std::endl;
}
