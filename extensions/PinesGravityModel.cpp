// DRAFT: performance-oriented structure for spherical-harmonic-style evaluation
// Keep your existing math/indices; swap in these patterns in the hot path.

#include <cmath>
#include <vector>
#include <algorithm>

// Example helper: fast integer power by recurrence (avoid std::pow in loops)
static inline void buildInvRPower(double invR, int maxN, std::vector<double>& invRpow) {
    invRpow.resize(maxN + 1);
    invRpow[0] = 1.0;
    for (int n = 1; n <= maxN; ++n) {
        invRpow[n] = invRpow[n - 1] * invR;
    }
}

// Example helper: sin(m*lambda), cos(m*lambda) via recurrence
static inline void buildTrigRecurrence(double lambda, int maxM,
                                       std::vector<double>& cosML,
                                       std::vector<double>& sinML) {
    cosML.resize(maxM + 1);
    sinML.resize(maxM + 1);

    const double c = std::cos(lambda);
    const double s = std::sin(lambda);

    cosML[0] = 1.0;
    sinML[0] = 0.0;

    // m=1
    if (maxM >= 1) {
        cosML[1] = c;
        sinML[1] = s;
    }

    // recurrence:
    // cos((m+1)l)=cos(ml)c-sin(ml)s
    // sin((m+1)l)=sin(ml)c+cos(ml)s
    for (int m = 1; m < maxM; ++m) {
        cosML[m + 1] = cosML[m] * c - sinML[m] * s;
        sinML[m + 1] = sinML[m] * c + cosML[m] * s;
    }
}

// Sketch of optimized hot function shape
// Replace placeholders with your exact data members and equations.
void EvaluateGravityFastDraft(
    const double x, const double y, const double z,
    const int Nmax,
    const int Mmax,
    const std::vector<double>& Cnm,   // flattened [n,m] indexing
    const std::vector<double>& Snm,
    double& ax, double& ay, double& az)
{
    // --- invariants ---
    const double r2 = x*x + y*y + z*z;
    const double r  = std::sqrt(r2);
    const double invR = 1.0 / r;

    // longitude-like angle once
    const double lambda = std::atan2(y, x);

    // allocate once per object in production; shown local for draft clarity
    std::vector<double> invRpow;
    std::vector<double> cosML, sinML;

    buildInvRPower(invR, Nmax + 2, invRpow);
    buildTrigRecurrence(lambda, Mmax, cosML, sinML);

    // TODO: compute associated Legendre terms with recurrence and preallocated arrays
    // Avoid per-iteration temporaries and function calls in inner loops.

    // --- accumulation ---
    double gx = 0.0, gy = 0.0, gz = 0.0;

    // Example loop skeleton only:
    for (int n = 0; n <= Nmax; ++n) {
        const double rn = invRpow[n + 1]; // pattern; adjust to your formulation
        const int mLimit = std::min(n, Mmax);

        for (int m = 0; m <= mLimit; ++m) {
            // flatten index helper (replace with your real indexing)
            const int idx = n*(n+1)/2 + m;

            const double c = Cnm[idx];
            const double s = Snm[idx];

            // combine harmonic phase once
            const double phase = c * cosML[m] + s * sinML[m];

            // TODO: multiply by your Pnm / derivatives from recurrence buffers
            // and accumulate gx, gy, gz according to your exact equations.
            const double term = rn * phase;

            // placeholders:
            gx += term * x;
            gy += term * y;
            gz += term * z;
        }
    }

    // final scaling per your model constants
    ax = gx;
    ay = gy;
    az = gz;
}
