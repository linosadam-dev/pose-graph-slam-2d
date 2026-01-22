# pose-graph-slam-2d

A from-scratch C++ implementation of a **2D pose graph SLAM backend (SE(2))**, built to demonstrate core SLAM mechanics without relying on external libraries.

This project focuses on the *backend* of SLAM: pose graph formulation, residuals, Jacobians, and nonlinear optimization. It intentionally avoids frameworks like g2o or Ceres to make the underlying math and algorithms explicit and easy to reason about.

---

## Overview

The system simulates a robot moving through a planar environment using noisy odometry, accumulates drift, and then corrects that drift using a loop closure constraint and **Gauss–Newton optimization**.

The implementation includes:

- SE(2) pose representations and operators (⊕, ⊖)
- Pose graph edges (odometry + loop closure)
- Residual and cost formulation
- Numeric Jacobians (finite differences)
- Gauss–Newton optimiser
- Gauge freedom handling (anchoring the first pose)
- Robust loss (Huber) for loop closures
- Quantitative evaluation metrics
- CSV export for trajectory visualisation

The goal is clarity and correctness rather than raw performance.

---

## How it works (high level)

1. **Ground truth trajectory**  
   A closed-loop trajectory is generated using a known motion model.

2. **Noisy odometry**  
   Gaussian noise is added to relative motions, and the robot’s estimated trajectory is built by chaining these noisy measurements. This causes drift.

3. **Pose graph construction**  
   - Odometry edges connect consecutive poses  
   - A loop closure edge connects the final pose back to the start using ground truth

4. **Optimisation**  
   The pose graph is optimised using Gauss–Newton:
   - Residuals are computed in SE(2)
   - Jacobians are computed numerically
   - Normal equations are assembled and solved
   - The first pose is fixed to remove gauge freedom

5. **Evaluation**  
   The system reports:
   - Loop closure residuals before and after optimisation
   - Total cost before and after optimisation
   - RMSE position error
   - Mean absolute heading error

---

## Expected behavior

When you run the program, you should see:

- A noticeable drift in the final pose **before** optimisation
- A large loop closure residual before optimisation
- Rapid convergence of Gauss–Newton (cost drops by orders of magnitude)
- A much smaller loop closure residual **after** optimisation
- Significantly improved RMSE and heading error

This demonstrates the core value of pose graph SLAM: **global consistency through optimisation**.

---

## How to run

This project is a single self-contained C++ file.

```bash
g++ main.cpp -std=c++17 -O2
./a.out
