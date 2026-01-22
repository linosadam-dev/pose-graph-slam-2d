#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <limits>
#include <iomanip>

// ============================================================
// 2D Pose Graph SLAM (SE(2)) - From scratch in one file
// Features:
// - SE(2) pose ops: compose (⊕), inverse, between (⊖)
// - Pose graph edges (odometry + loop closure)
// - Residuals and weighted cost
// - Robust Huber loss (optional for loop closures)
// - Numeric Jacobians (finite differences)
// - Gauss-Newton optimizer with gauge fixing (anchor pose 0)
// - Simple Gaussian elimination linear solver
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
    // e is sqrt(weighted_error), delta is threshold
    if (e <= delta) return 1.0;
    return delta / e;
}

// ============================================================
// Linear algebra utilities (library-free)
// We'll use dense matrices/vectors stored as std::vector<double>
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

// Simple Gaussian elimination with partial pivoting
// Solves A x = b for x. Returns false if singular.
bool solveLinearSystem(DenseMat A, DenseVec b, DenseVec& x_out) {
    const int n = A.n;
    x_out = DenseVec(n);

    // Forward elimination
    for (int k = 0; k < n; k++) {
        // Pivot
        int piv = k;
        double maxAbs = std::fabs(A(k,k));
        for (int r = k+1; r < n; r++) {
            double val = std::fabs(A(r,k));
            if (val > maxAbs) { maxAbs = val; piv = r; }
        }
        if (maxAbs < 1e-14) return false; // singular / ill-conditioned

        // Swap rows in A and b
        if (piv != k) {
            for (int c = k; c < n; c++) std::swap(A(k,c), A(piv,c));
            std::swap(b[k], b[piv]);
        }

        // Eliminate below
        for (int r = k+1; r < n; r++) {
            double f = A(r,k) / A(k,k);
            A(r,k) = 0.0;
            for (int c = k+1; c < n; c++) {
                A(r,c) -= f * A(k,c);
            }
            b[r] -= f * b[k];
        }
    }

    // Back substitution
    for (int i = n-1; i >= 0; i--) {
        double s = b[i];
        for (int c = i+1; c < n; c++) s -= A(i,c) * x_out[c];
        x_out[i] = s / A(i,i);
    }

    return true;
}

// ============================================================
// Numeric Jacobians for one edge
// We need A = dr/dxi (3x3) and B = dr/dxj (3x3)
// We'll compute by finite difference on box-plus.
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
// Build and solve Gauss-Newton normal equations
// ============================================================

// Add 3x3 block into H at (bi,bj)
void addBlock(DenseMat& H, int bi, int bj, const double blk[3][3]) {
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            H(bi + r, bj + c) += blk[r][c];
        }
    }
}

// Add 3-vector into g at index bi
void addVec(DenseVec& g, int bi, const double v3[3]) {
    for (int r = 0; r < 3; r++) g[bi + r] += v3[r];
}

// Multiply 3x3^T * W(3x3 diag) * 3x3  -> 3x3
// Here W is diag([w_t, w_t, w_r]) * robust_weight
void compute_AT_W_A(const Mat3& A, double w_t, double w_r, double robust_w, double out[3][3]) {
    // W diagonal
    double W[3] = { robust_w * w_t, robust_w * w_t, robust_w * w_r };

    // out = A^T * diag(W) * A
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            out[r][c] = 0.0;

    for (int k = 0; k < 3; k++) { // row in residual space
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                out[r][c] += A.m[k][r] * W[k] * A.m[k][c];
            }
        }
    }
}

// Compute A^T * W * B -> 3x3
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

// Compute A^T * W * r -> 3
void compute_AT_W_r(const Mat3& A, const std::vector<double>& r, double w_t, double w_r, double robust_w, double out[3]) {
    double W[3] = { robust_w * w_t, robust_w * w_t, robust_w * w_r };
    out[0]=out[1]=out[2]=0.0;

    for (int k = 0; k < 3; k++) {
        for (int c = 0; c < 3; c++) {
            out[c] += A.m[k][c] * W[k] * r[k];
        }
    }
}

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

        // Build H and g
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

            // Blocks:
            // H_ii += A^T W A
            // H_ij += A^T W B
            // H_ji += B^T W A
            // H_jj += B^T W B
            // g_i  += A^T W r
            // g_j  += B^T W r

            double Hii[3][3], Hij[3][3], Hjj[3][3];
            compute_AT_W_A(A, e.w_t, e.w_r, robust_w, Hii);
            compute_AT_W_B(A, B, e.w_t, e.w_r, robust_w, Hij);
            compute_AT_W_A(B, e.w_t, e.w_r, robust_w, Hjj);

            // B^T W A is transpose of A^T W B if W is diagonal (it is)
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

        // Gauge fixing: anchor pose 0 strongly (x0,y0,theta0)
        // Add a huge prior on the first 3 variables: delta0 = 0
        const double lambda = 1e12;
        H(0,0) += lambda;
        H(1,1) += lambda;
        H(2,2) += lambda;

        // Solve H * dx = -g
        DenseVec b(dim);
        for (int i = 0; i < dim; i++) b[i] = -g[i];

        DenseVec dx;
        bool ok = solveLinearSystem(H, b, dx);
        if (!ok) {
            std::cout << "Linear solve failed (singular). Try different settings.\n";
            return false;
        }

        // Apply update
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

        // Convergence check
        if (max_step < 1e-6) break;

        // If cost increased badly (rare with GN), you can stop early
        // This keeps the demo stable in basic environments.
        if (new_cost > cost * 1.5) {
            std::cout << "Warning: cost increased significantly; stopping early.\n";
            break;
        }
    }

    return true;
}

// ============================================================
// Simulation helpers
// ============================================================

double randn(std::mt19937& rng, double sigma) {
    std::normal_distribution<double> dist(0.0, sigma);
    return dist(rng);
}

int main() {
    // ----- 1) Ground truth trajectory -----
    const int N = 40;
    Pose2D u_true(0.5, 0.0, PI/20.0); // forward + small turn => curved path

    std::vector<Pose2D> gt;
    gt.reserve(N+1);
    gt.push_back(Pose2D(0,0,0));
    for (int k = 0; k < N; k++) {
        gt.push_back(compose(gt.back(), u_true));
    }

    // ----- 2) Noisy odometry measurements + drifting initial estimate -----
    std::mt19937 rng(42);
    const double sigma_trans = 0.02;              // meters
    const double sigma_rot   = (1.0 * PI/180.0);  // 1 deg in rad

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

    // Print before optimization
    std::cout << "N = " << N << "\n";
    std::cout << "Initial estimate start: x0=(" << x[0].x << "," << x[0].y << "," << x[0].theta << ")\n";
    std::cout << "Initial estimate end:   xN=(" << x[N].x << "," << x[N].y << "," << x[N].theta << ")\n";
    std::cout << "Ground truth end:       gtN=(" << gt[N].x << "," << gt[N].y << "," << gt[N].theta << ")\n";

    auto r_loop0 = residual(x[0], x[N], z_loop_true);
    std::cout << "\nLoop closure residual BEFORE:\n";
    std::cout << "r_loop = [" << r_loop0[0] << ", " << r_loop0[1] << ", " << r_loop0[2] << "]\n";

    double cost0 = totalCost(edges, x, true, 1.0);
    std::cout << "Total cost (before): " << cost0 << "\n\n";

    // ----- 4) Optimize -----
    std::cout << "Running Gauss-Newton...\n";
    gaussNewtonOptimize(x, edges, 10, true, 1.0);

    // Print after optimization
    auto r_loop1 = residual(x[0], x[N], z_loop_true);
    std::cout << "\nLoop closure residual AFTER:\n";
    std::cout << "r_loop = [" << r_loop1[0] << ", " << r_loop1[1] << ", " << r_loop1[2] << "]\n";

    double cost1 = totalCost(edges, x, true, 1.0);
    std::cout << "Total cost (after): " << cost1 << "\n";

    std::cout << "\nOptimized end pose: xN=(" << x[N].x << "," << x[N].y << "," << x[N].theta << ")\n";
    std::cout << "Ground truth end:   gtN=(" << gt[N].x << "," << gt[N].y << "," << gt[N].theta << ")\n";

    return 0;
}
