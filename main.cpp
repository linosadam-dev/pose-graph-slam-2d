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

// --- Pose graph bits ---

struct Edge {
    int i;       // from node i
    int j;       // to node j
    Pose2D z;    // measured relative transform z_ij (i -> j) in frame i
    double w_t;  // translation weight
    double w_r;  // rotation weight
    bool is_loop;

    Edge(int i_, int j_, const Pose2D& z_, double wt=1.0, double wr=1.0, bool loop=false)
        : i(i_), j(j_), z(z_), w_t(wt), w_r(wr), is_loop(loop) {}
};

// Residual r_ij = predicted - measured (angle wrapped)
std::vector<double> residual(const Pose2D& xi, const Pose2D& xj, const Pose2D& z) {
    Pose2D zhat = between(xi, xj);

    std::vector<double> r(3);
    r[0] = zhat.x - z.x;
    r[1] = zhat.y - z.y;
    r[2] = wrapAngle(zhat.theta - z.theta);
    return r;
}

// Weighted squared error for one edge
double edgeCost(const Edge& e, const std::vector<Pose2D>& x) {
    auto r = residual(x[e.i], x[e.j], e.z);
    return e.w_t * (r[0]*r[0] + r[1]*r[1]) + e.w_r * (r[2]*r[2]);
}

// Total graph cost
double totalCost(const std::vector<Edge>& edges, const std::vector<Pose2D>& x) {
    double cost = 0.0;
    for (const auto& e : edges) cost += edgeCost(e, x);
    return cost;
}

int main() {
    // Build a simple "square" trajectory estimate by chaining odometry motions
    std::vector<Pose2D> x;
    x.push_back(Pose2D(0, 0, 0));

    Pose2D u(1.0, 0.0, PI/2.0); // odometry motion each step

    for (int k = 0; k < 4; k++) {
        x.push_back(compose(x.back(), u));
    }

    std::cout << "Initial trajectory (estimate):\n";
    for (size_t i = 0; i < x.size(); i++) {
        std::cout << i << ": x=" << x[i].x << ", y=" << x[i].y << ", theta=" << x[i].theta << "\n";
    }

    // Build pose graph edges
    // Odometry edges: (k -> k+1)
    std::vector<Edge> edges;
    for (int k = 0; k < 4; k++) {
        edges.emplace_back(k, k+1, u, 1.0, 1.0, false);
    }

    // Add a loop closure: last pose back to start should be identity
    // In perfect world: x4 should equal x0
    Pose2D z_loop(0.0, 0.0, 0.0);
    edges.emplace_back(0, 4, z_loop, 1.0, 1.0, true);

    // Print residuals
    std::cout << "\nEdge residuals:\n";
    for (const auto& e : edges) {
        auto r = residual(x[e.i], x[e.j], e.z);
        std::cout << (e.is_loop ? "[LOOP] " : "[ODOM] ")
                  << e.i << " -> " << e.j
                  << "  r = [" << r[0] << ", " << r[1] << ", " << r[2] << "]\n";
    }

    std::cout << "\nTotal cost: " << totalCost(edges, x) << "\n";

    return 0;
}
