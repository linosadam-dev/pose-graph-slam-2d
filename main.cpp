#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <limits>
#include <iomanip>
#include <fstream>
#include <string>

// ============================================================
// 2D Pose Graph SLAM (SE(2)) - From scratch in one file
// Features:
// - SE(2) pose ops: compose (⊕), inverse, between (⊖)
// - Pose graph edges (odometry + loop closure)
// - Residuals and weighted cost
// - Robust Huber loss (for loop closures)
// - Numeric Jacobians (finite differences)
// - Gauss-Newton optimizer with gauge fixing (anchor pose 0)
// - Simple Gaussian elimination linear solver (dense)
// - CSV export: traj_gt.csv, traj_before.csv, traj_after.csv
// - Metrics: RMSE position + mean abs heading error
// ============================================================

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

// SE(2) compose: a ⊕ b
Pose2D compose(const Pose2D& a, const Pose2D& b) {
    double c = std::cos(a.theta);
    double s = std::sin(a.theta);

    Pose2D r;
    r.x = a.x + c * b.x - s * b.y;
    r.y = a.y + s * b.x + c * b.y;
    r.theta = wrapAngle(a.theta + b.theta);
    return r;
}

// SE(2) inverse
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

// "Box-plus" update on SE(2): x <- x ⊕ dx
Pose2D boxPlus(const Pose2D& x, const Pose2D& dx) {
    return compose(x, dx);
}

// ============================================================
// Pose graph
// ============================================================

struct Edge {
    int i;
    int j;
    Pose2D z;     // measurement z_ij (i->j) in frame i
    double w_t;   // translation weight
    double w_r;   // rotation weight
    bool is_loop;

    Edge(int i_, int j_, const Pose2D& z_, double wt=1.0, double wr=1.0, bool loop=false)
        : i(i_), j(j_), z(z_), w_t(wt), w_r(wr), is_loop(loop) {}
};

// Residual r = predicted - measured (angle wrapped)
std::vector<double> residual(const Pose2D& xi, const Pose2D& xj, const Pose2D& z) {
    Pose2D zhat = between(xi, xj);
    std::vector<double> r(3);
    r[0] = zhat.x - z.x;
    r[1] = zhat.y - z.y;
    r[2] = wrapAngle(zhat.theta - z.theta);
    return r;
}

// Weighted squared error for a residual vector
double weightedError(const std::vector<double>& r, double w_t, double w_r) {
    return w_t * (r[0]*r[0] + r[1]*r[1]) + w_r * (r[2]*r[2]);
}

// Huber robust weight based on sqrt(error)
// Returns a multiplier in (0,1] to downweight large residuals
double huberWeightFromError(double e, double delta) {
    if (e <= delta) return 1.0;
    return delta / e;
}

// ============================================================
// Linear algebra utilities (dense, no external libs)
// ============================================================

struct DenseMat {
    int n; // n x n
    std::vector<double> a; // row-major size n*n
    DenseMat(int n_=0) : n(n_), a(n_*n_, 0.0) {}

    double& operator()(int r, int c) { return a[r*n + c]; }
    double  operator()(int r, int c) const { return a[r*n + c]; }
};

struct DenseVec {
    std::vector<double> v;
    DenseVec(int n=0) : v(n, 0.0) {}
    int size() const { return (int)v.size(); }
    double& operator[](int i) { return v[i]; }
    double  operator[](int i) const { return v[i]; }
};

// Gaussian elimination with partial pivoting
bool solveLinearSystem(DenseMat A, DenseVec b, DenseVec& x_out) {
    const int n = A.n;
    x_out = DenseVec(n);

    for (int k = 0; k < n; k++) {
        int piv = k;
        double maxAbs = std::fabs(A(k,k));
        for (int r = k+1; r < n; r++) {
            double val = std::fabs(A(r,k));
            if (val > maxAbs) { maxAbs = val; piv = r; }
        }
        if (maxAbs < 1e-14) return false;

        if (piv != k) {
            for (int c = k; c < n; c++) std::swap(A(k,c), A(piv,c));
            std::swap(b[k], b[piv]);
        }

        for (int r = k+1; r < n; r++) {
            double f = A(r,k) / A(k,k);
            A(r,k) = 0.0;
            for (int c = k+1; c < n; c++) {
                A(r,c) -= f * A(k,c);
            }
            b[r] -= f * b[k];
        }
    }

    for (int i = n-1; i >= 0; i--) {
        double s = b[i];
        for (int c = i+1; c < n; c++) s -= A(i,c) * x_out[c];
        x_out[i] = s / A(i,i);
    }
    return true;
}

// ============================================================
// Numeric Jacobians (finite differences)
// ============================================================

struct Mat3 {
    double m[3][3];
};

Mat3 numericJacobian_wrt_i(const Pose2D& xi, const Pose2D& xj, const Pose2D& z,
                           double eps = 1e-6) {
    Mat3 J{};
    auto r0 = residual(xi, xj, z);

    for (int col = 0; col < 3; col++) {
        Pose2D d(0,0,0);
        if (col == 0) d.x = eps;
        if (col == 1) d.y = eps;
        if (col == 2) d.theta = eps;

        Pose2D xi_pert = boxPlus(xi, d);
        auto r1 = residual(xi_pert, xj, z);

        for (int row = 0; row < 3; row++) {
            J.m[row][col] = (r1[row] - r0[row]) / eps;
        }
    }
    return J;
}

Mat3 numericJacobian_wrt_j(const Pose2D& xi, const Pose2D& xj, const Pose2D& z,
                           double eps = 1e-6) {
    Mat3 J{};
    auto r0 = residual(xi, xj, z);

    for (int col = 0; col < 3; col++) {
        Pose2D d(0,0,0);
        if (col == 0) d.x = eps;
        if (col == 1) d.y = eps;
        if (col == 2) d.theta = eps;

        Pose2D xj_pert = boxPlus(xj, d);
        auto r1 = residual(xi, xj_pert, z);

        for (int row = 0; row < 3; row++) {
            J.m[row][col] = (r1[row] - r0[row]) / eps;
        }
    }
    return J;
}

// ============================================================
// Helpers to assemble normal equations
// ============================================================

void addBlock(DenseMat& H, int bi, int bj, const double blk[3][3]) {
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            H(bi + r, bj + c) += blk[r][c];
        }
    }
}

void addVec(DenseVec& g, int bi, const double v3[3]) {
    for (int r = 0; r < 3; r++) g[bi + r] += v3[r];
}

// out = A^T * diag(W) * A, where W = [w_t, w_t, w_r] * robust_w
void compute_AT_W_A(const Mat3& A, double w_t, double w_r, double robust_w, double out[3][3]) {
    double W[3] = { robust_w * w_t, robust_w * w_t, robust_w * w_r };

    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            out[r][c] = 0.0;

    for (int k = 0; k < 3; k++) {
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                out[r][c] += A.m[k][r] * W[k] * A.m[k][c];
            }
        }
    }
}

// out = A^T * diag(W) * B
void compute_AT_W_B(const Mat3& A, const Mat3& B, double w_t, double w_r, double robust_w, double out[3][3]) {
    double W[3] = { robust_w * w_t, robust_w * w_t, robust_w * w_r };

    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            out[r][c] = 0.0;

    for (int k = 0; k < 3; k++) {
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                out[r][c] += A.m[k][r] * W[k] * B.m[k][c];
            }
        }
    }
}

// out = A^T * diag(W) * r
void compute_AT_W_r(const Mat3& A, const std::vector<double>& r, double w_t, double w_r, double robust_w, double out[3]) {
    double W[3] = { robust_w * w_t, robust_w * w_t, robust_w * w_r };
    out[0]=out[1]=out[2]=0.0;

    for (int k = 0; k < 3; k++) {
        for (int c = 0; c < 3; c++) {
            out[c] += A.m[k][c] * W[k] * r[k];
        }
    }
}

// ============================================================
// Cost and optimizer
// ============================================================

double totalCost(const std::vector<Edge>& edges, const std::vector<Pose2D>& x,
                 bool use_huber=false, double huber_delta=1.0) {
    double cost = 0.0;
    for (const auto& e : edges) {
        auto r = residual(x[e.i], x[e.j], e.z);
        double err = weightedError(r, e.w_t, e.w_r);

        double robust_w = 1.0;
        if (use_huber && e.is_loop) {
            robust_w = huberWeightFromError(std::sqrt(err), huber_delta);
        }
        cost += robust_w * err;
    }
    return cost;
}

bool gaussNewtonOptimize(std::vector<Pose2D>& x, const std::vector<Edge>& edges,
                         int max_iters=10, bool use_huber=true, double huber_delta=1.0) {
    const int N = (int)x.size();
    const int dim = 3 * N;

    for (int iter = 0; iter < max_iters; iter++) {
        DenseMat H(dim);
        DenseVec g(dim);

        double cost = 0.0;

        for (const auto& e : edges) {
            const Pose2D& xi = x[e.i];
            const Pose2D& xj = x[e.j];

            auto r = residual(xi, xj, e.z);
            double err = weightedError(r, e.w_t, e.w_r);

            double robust_w = 1.0;
            if (use_huber && e.is_loop) {
                robust_w = huberWeightFromError(std::sqrt(err), huber_delta);
            }
            cost += robust_w * err;

            Mat3 A = numericJacobian_wrt_i(xi, xj, e.z);
            Mat3 B = numericJacobian_wrt_j(xi, xj, e.z);

            double Hii[3][3], Hij[3][3], Hjj[3][3];
            compute_AT_W_A(A, e.w_t, e.w_r, robust_w, Hii);
            compute_AT_W_B(A, B, e.w_t, e.w_r, robust_w, Hij);
            compute_AT_W_A(B, e.w_t, e.w_r, robust_w, Hjj);

            double Hji[3][3];
            for (int r0=0; r0<3; r0++)
                for (int c0=0; c0<3; c0++)
                    Hji[r0][c0] = Hij[c0][r0];

            double gi[3], gj[3];
            compute_AT_W_r(A, r, e.w_t, e.w_r, robust_w, gi);
            compute_AT_W_r(B, r, e.w_t, e.w_r, robust_w, gj);

            int bi = 3 * e.i;
            int bj = 3 * e.j;

            addBlock(H, bi, bi, Hii);
            addBlock(H, bi, bj, Hij);
            addBlock(H, bj, bi, Hji);
            addBlock(H, bj, bj, Hjj);

            addVec(g, bi, gi);
            addVec(g, bj, gj);
        }

        // Gauge fixing: anchor pose 0 with a strong prior (delta = 0)
        const double lambda = 1e12;
        H(0,0) += lambda;
        H(1,1) += lambda;
        H(2,2) += lambda;

        DenseVec b(dim);
        for (int i = 0; i < dim; i++) b[i] = -g[i];

        DenseVec dx;
        if (!solveLinearSystem(H, b, dx)) {
            std::cout << "Linear solve failed (singular/ill-conditioned).\n";
            return false;
        }

        double max_step = 0.0;
        for (int k = 0; k < N; k++) {
            Pose2D d(dx[3*k + 0], dx[3*k + 1], dx[3*k + 2]);
            x[k] = boxPlus(x[k], d);

            max_step = std::max(max_step, std::fabs(dx[3*k + 0]));
            max_step = std::max(max_step, std::fabs(dx[3*k + 1]));
            max_step = std::max(max_step, std::fabs(dx[3*k + 2]));
        }

        double new_cost = totalCost(edges, x, use_huber, huber_delta);

        std::cout << "Iter " << iter
                  << "  cost=" << std::setprecision(10) << cost
                  << "  new_cost=" << new_cost
                  << "  max_step=" << max_step
                  << "\n";

        if (max_step < 1e-6) break;

        if (new_cost > cost * 1.5) {
            std::cout << "Warning: cost increased significantly; stopping early.\n";
            break;
        }
    }

    return true;
}

// ============================================================
// Simulation + evaluation
// ============================================================

double randn(std::mt19937& rng, double sigma) {
    std::normal_distribution<double> dist(0.0, sigma);
    return dist(rng);
}

void writeCSV(const std::string& filename, const std::vector<Pose2D>& traj) {
    std::ofstream f(filename);
    if (!f) {
        std::cout << "Warning: could not write file: " << filename << "\n";
        return;
    }
    f << "idx,x,y,theta\n";
    for (size_t i = 0; i < traj.size(); i++) {
        f << i << "," << traj[i].x << "," << traj[i].y << "," << traj[i].theta << "\n";
    }
}

struct Metrics {
    double rmse_pos;
    double mean_abs_theta;
};

Metrics computeMetrics(const std::vector<Pose2D>& gt, const std::vector<Pose2D>& est) {
    const size_t n = std::min(gt.size(), est.size());
    double sum_sq = 0.0;
    double sum_abs_th = 0.0;

    for (size_t i = 0; i < n; i++) {
        double dx = est[i].x - gt[i].x;
        double dy = est[i].y - gt[i].y;
        sum_sq += dx*dx + dy*dy;

        double dth = wrapAngle(est[i].theta - gt[i].theta);
        sum_abs_th += std::fabs(dth);
    }

    Metrics m;
    m.rmse_pos = std::sqrt(sum_sq / (double)n);
    m.mean_abs_theta = sum_abs_th / (double)n;
    return m;
}

int main() {
    // ----- 1) Ground truth trajectory -----
    const int N = 40;
    Pose2D u_true(0.5, 0.0, PI/20.0); // forward + small turn => curved path (closes over 40 steps)

    std::vector<Pose2D> gt;
    gt.reserve(N+1);
    gt.push_back(Pose2D(0,0,0));
    for (int k = 0; k < N; k++) {
        gt.push_back(compose(gt.back(), u_true));
    }

    // ----- 2) Noisy odometry measurements + drifting initial estimate -----
    std::mt19937 rng(42);                  // fixed seed (reproducible)
    const double sigma_trans = 0.02;       // meters
    const double sigma_rot   = (1.0 * PI/180.0); // 1 degree in radians

    std::vector<Edge> edges;
    edges.reserve(N + 1);

    std::vector<Pose2D> x; // initial estimate
    x.reserve(N+1);
    x.push_back(Pose2D(0,0,0));

    for (int k = 0; k < N; k++) {
        Pose2D z_true = between(gt[k], gt[k+1]);

        Pose2D z_noisy(
            z_true.x + randn(rng, sigma_trans),
            z_true.y + randn(rng, sigma_trans),
            wrapAngle(z_true.theta + randn(rng, sigma_rot))
        );

        // Odometry edges
        edges.emplace_back(k, k+1, z_noisy, 1.0, 1.0, false);

        // Chain odometry to get drifting initial estimate
        x.push_back(compose(x.back(), z_noisy));
    }

    // ----- 3) Loop closure (ground truth constraint) -----
    Pose2D z_loop_true = between(gt[0], gt[N]);
    edges.emplace_back(0, N, z_loop_true, 5.0, 5.0, true);

    // Save BEFORE (for CSV + metrics)
    std::vector<Pose2D> x_before = x;

    std::cout << "N = " << N << "\n";
    std::cout << "Initial estimate start: x0=(" << x[0].x << "," << x[0].y << "," << x[0].theta << ")\n";
    std::cout << "Initial estimate end:   xN=(" << x[N].x << "," << x[N].y << "," << x[N].theta << ")\n";
    std::cout << "Ground truth end:       gtN=(" << gt[N].x << "," << gt[N].y << "," << gt[N].theta << ")\n";

    auto r_loop0 = residual(x[0], x[N], z_loop_true);
    std::cout << "\nLoop closure residual BEFORE:\n";
    std::cout << "r_loop = [" << r_loop0[0] << ", " << r_loop0[1] << ", " << r_loop0[2] << "]\n";

    double cost0 = totalCost(edges, x, true, 1.0);
    std::cout << "Total cost (before): " << cost0 << "\n";

    auto mb = computeMetrics(gt, x_before);
    std::cout << "\nMetrics BEFORE:\n";
    std::cout << "RMSE position: " << mb.rmse_pos << "\n";
    std::cout << "Mean abs heading error: " << mb.mean_abs_theta << "\n";

    // CSV exports (may not work in some online compilers)
    writeCSV("traj_gt.csv", gt);
    writeCSV("traj_before.csv", x_before);

    // ----- 4) Optimize -----
    std::cout << "\nRunning Gauss-Newton...\n";
    gaussNewtonOptimize(x, edges, 10, true, 1.0);

    auto r_loop1 = residual(x[0], x[N], z_loop_true);
    std::cout << "\nLoop closure residual AFTER:\n";
    std::cout << "r_loop = [" << r_loop1[0] << ", " << r_loop1[1] << ", " << r_loop1[2] << "]\n";

    double cost1 = totalCost(edges, x, true, 1.0);
    std::cout << "Total cost (after): " << cost1 << "\n";

    auto ma = computeMetrics(gt, x);
    std::cout << "\nMetrics AFTER:\n";
    std::cout << "RMSE position: " << ma.rmse_pos << "\n";
    std::cout << "Mean abs heading error: " << ma.mean_abs_theta << "\n";

    std::cout << "\nOptimized end pose: xN=(" << x[N].x << "," << x[N].y << "," << x[N].theta << ")\n";
    std::cout << "Ground truth end:   gtN=(" << gt[N].x << "," << gt[N].y << "," << gt[N].theta << ")\n";

    writeCSV("traj_after.csv", x);

    std::cout << "\nWrote CSV files (if supported): traj_gt.csv, traj_before.csv, traj_after.csv\n";

    return 0;
}
