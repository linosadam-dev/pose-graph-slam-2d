#include <iostream>
#include <vector>
#include <cmath>
#include <random>

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
    int i;
    int j;
    Pose2D z;     // measured relative transform
    double w_t;   // translation weight
    double w_r;   // rotation weight
    bool is_loop;

    Edge(int i_, int j_, const Pose2D& z_, double wt=1.0, double wr=1.0, bool loop=false)
        : i(i_), j(j_), z(z_), w_t(wt), w_r(wr), is_loop(loop) {}
};

std::vector<double> residual(const Pose2D& xi, const Pose2D& xj, const Pose2D& z) {
    Pose2D zhat = between(xi, xj);

    std::vector<double> r(3);
    r[0] = zhat.x - z.x;
    r[1] = zhat.y - z.y;
    r[2] = wrapAngle(zhat.theta - z.theta);
    return r;
}

double edgeCost(const Edge& e, const std::vector<Pose2D>& x) {
    auto r = residual(x[e.i], x[e.j], e.z);
    return e.w_t * (r[0]*r[0] + r[1]*r[1]) + e.w_r * (r[2]*r[2]);
}

double totalCost(const std::vector<Edge>& edges, const std::vector<Pose2D>& x) {
    double cost = 0.0;
    for (const auto& e : edges) cost += edgeCost(e, x);
    return cost;
}

// Gaussian noise helper
double randn(std::mt19937& rng, double sigma) {
    std::normal_distribution<double> dist(0.0, sigma);
    return dist(rng);
}

int main() {
    // ----- 1) Ground truth trajectory -----
    // Make a longer path so drift is visible
    const int N = 40;  // number of steps
    Pose2D u_true(0.5, 0.0, PI/20.0); // small forward + small turn each step (curving path)

    std::vector<Pose2D> gt;
    gt.reserve(N+1);
    gt.push_back(Pose2D(0,0,0));
    for (int k = 0; k < N; k++) {
        gt.push_back(compose(gt.back(), u_true));
    }

    // ----- 2) Noisy odometry measurements -----
    std::mt19937 rng(42); // fixed seed for repeatability
    const double sigma_trans = 0.02;         // meters
    const double sigma_rot   = (1.0 * PI/180.0); // 1 degree in radians

    std::vector<Edge> edges;
    edges.reserve(N + 1);

    // Build initial estimate by chaining noisy odometry
    std::vector<Pose2D> x;
    x.reserve(N+1);
    x.push_back(Pose2D(0,0,0));

    for (int k = 0; k < N; k++) {
        // Measurement z_k is the true relative motion + noise (in frame of gt[k])
        Pose2D z_true = between(gt[k], gt[k+1]);

        Pose2D z_noisy(
            z_true.x + randn(rng, sigma_trans),
            z_true.y + randn(rng, sigma_trans),
            wrapAngle(z_true.theta + randn(rng, sigma_rot))
        );

        edges.emplace_back(k, k+1, z_noisy, 1.0, 1.0, false);

        // Chain the noisy measurements to get an initial drifting estimate
        x.push_back(compose(x.back(), z_noisy));
    }

    // ----- 3) Add a loop closure edge (truth) -----
    // Connect last pose back to start using ground truth relative transform
    Pose2D z_loop_true = between(gt[0], gt[N]); // where the last should be relative to start
    edges.emplace_back(0, N, z_loop_true, 5.0, 5.0, true); // heavier weight than odom

    // ----- 4) Print summary -----
    std::cout << "N = " << N << "\n";
    std::cout << "Initial estimate start: x0=(" << x[0].x << "," << x[0].y << "," << x[0].theta << ")\n";
    std::cout << "Initial estimate end:   xN=(" << x[N].x << "," << x[N].y << "," << x[N].theta << ")\n";
    std::cout << "Ground truth end:       gtN=(" << gt[N].x << "," << gt[N].y << "," << gt[N].theta << ")\n";

    // Show loop closure residual (this should now be NOT small)
    auto r_loop = residual(x[0], x[N], z_loop_true);
    std::cout << "\nLoop closure residual (should be noticeable due to drift):\n";
    std::cout << "r_loop = [" << r_loop[0] << ", " << r_loop[1] << ", " << r_loop[2] << "]\n";

    std::cout << "\nTotal cost (before optimization): " << totalCost(edges, x) << "\n";

    return 0;
}
