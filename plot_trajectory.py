import pandas as pd
import matplotlib.pyplot as plt

# Load trajectories
gt = pd.read_csv("traj_gt.csv")
before = pd.read_csv("traj_before.csv")
after = pd.read_csv("traj_after.csv")

# Plot
plt.figure(figsize=(6, 6))
plt.plot(gt.x, gt.y, 'k--', label="Ground truth")
plt.plot(before.x, before.y, 'r-', label="Before optimization")
plt.plot(after.x, after.y, 'g-', label="After optimization")

plt.axis("equal")
plt.grid(True)
plt.legend()
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("2D Pose Graph SLAM – Loop Closure Correction")

# Save + show
plt.savefig("trajectory.png", dpi=200)
plt.show()
