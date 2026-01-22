#include <iostream>
#include <vector>
#include <cmath>

const double PI = 3.141592653589793;

// Wrap angle to (-pi, pi]
double wrapAngle(double a) {
    while (a > PI)  a -= 2.0 * PI;
    while (a <= -PI) a += 2.0 * PI;
    return a;
}

struct Pose2D {
    double x, y, theta;

    Pose2D(double x_=0, double y_=0, double t_=0)
        : x(x_), y(y_), theta(wrapAngle(t_)) {}
};

// SE(2) pose composition: a ⊕ b
Pose2D compose(const Pose2D& a, const Pose2D& b) {
    double c = std::cos(a.theta);
    double s = std::sin(a.theta);

    Pose2D result;
    result.x = a.x + c * b.x - s * b.y;
    result.y = a.y + s * b.x + c * b.y;
    result.theta = wrapAngle(a.theta + b.theta);
    return result;
}

// SE(2) pose inverse
Pose2D inverse(const Pose2D& p) {
    double c = std::cos(p.theta);
    double s = std::sin(p.theta);

    Pose2D inv;
    inv.x = -c * p.x - s * p.y;
    inv.y =  s * p.x - c * p.y;
    inv.theta = wrapAngle(-p.theta);
    return inv;
}

// Relative pose: a ⊖ b = a^{-1} ⊕ b
Pose2D between(const Pose2D& a, const Pose2D& b) {
    return compose(inverse(a), b);
}

int main() {
    Pose2D pose(0, 0, 0);
    Pose2D motion(1.0, 0.0, PI / 2.0);

    std::vector<Pose2D> trajectory;
    trajectory.push_back(pose);

    for (int i = 0; i < 4; i++) {
        pose = compose(pose, motion);
        trajectory.push_back(pose);
    }

    std::cout << "Square trajectory:\n";
    for (size_t i = 0; i < trajectory.size(); i++) {
        const auto& p = trajectory[i];
        std::cout << i << ": x=" << p.x
                  << ", y=" << p.y
                  << ", theta=" << p.theta << "\n";
    }

    return 0;
}
