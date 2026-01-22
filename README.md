# pose-graph-slam-2d

A from-scratch C++ implementation of 2D pose graph SLAM (SE(2)), including simulation, nonlinear optimization (Gauss–Newton), robust loop closures, and trajectory evaluation.

## Overview
This project implements the **backend of a SLAM system** from first principles, focusing on:
- SE(2) pose representations and operators (⊕, ⊖)
- Pose graph residual formulation
- Nonlinear least squares optimization (Gauss–Newton)
- Robust loss functions for loop closures
- Gauge freedom handling and trajectory evaluation

The implementation is intentionally library-light to emphasize understanding of core SLAM mechanics.

## Features (planned / in progress)
- SE(2) pose composition, inversion, and relative pose
- Simulated robot trajectories with noisy odometry
- Pose graph construction (nodes + edges)
- Gauss–Newton optimization
- Robust loop closures (Huber loss + gating)
- CSV trajectory export and RMSE evaluation

## Motivation
This project was built to demonstrate fundamental SLAM competence and to serve as a foundation for future extensions (SE(3), sensor fusion, and multi-robot SLAM).

## Future extensions
- SE(3) pose graph SLAM
- IMU preintegration factors
- Multi-robot loop closures
- Robust SLAM under degraded sensing conditions (e.g. smoke)

---
