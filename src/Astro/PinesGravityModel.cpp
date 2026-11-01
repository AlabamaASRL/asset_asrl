
#include "PinesGravityModel.h"
#include "GravityCoefficientReader.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace ASSET {

namespace {

constexpr double sqrt_two = 1.4142135623730950488;

}  // namespace

// Construct a Pines spherical-harmonic gravity model.
//
// References:
//     S. Pines, "Uniform Representation of the Gravitational Potential
//     and its Derivatives," AIAA Journal, Vol. 11, No. 11, 1973,
//     pp. 1508-1511.
//     DOI: https://doi.org/10.2514/3.50619
//
// Coefficient-file parsing is handled by GravityCoefficientReader.
// This class is responsible for validating the model configuration,
// storing the coefficients, and precomputing the Pines recurrence factors.
PinesGravityModel::PinesGravityModel(
    const std::string& coefficient_file,
    int degree,
    double mu,
    double reference_radius,
    const std::string& normalization)
    : degree_(degree),
      mu_(mu),
      reference_radius_(reference_radius),
      coefficient_file_(coefficient_file) {

    // Validate the requested spherical-harmonic degree.
    if (degree_ < 0 || degree_ > 360) {
        throw std::invalid_argument(
            "PinesGravity degree must be in [0, 360]");
    }

    // Validate the fallback gravitational parameter and reference radius.
    // These values are used when the coefficient file has no metadata.
    if (!std::isfinite(mu_) || mu_ <= 0.0 ||
        !std::isfinite(reference_radius_) ||
        reference_radius_ <= 0.0) {
        throw std::invalid_argument(
            "PinesGravity mu and reference_radius must be finite and positive");
    }

    // This implementation uses fully normalized spherical-harmonic
    // coefficients. The argument remains available for coefficient files
    // that do not contain normalization metadata.
    if (normalization != "fully_normalized") {
        throw std::invalid_argument(
            "PinesGravity supports only fully_normalized coefficients");
    }

    // Read the coefficient file using the standalone coefficient reader.
    // The returned arrays use triangular storage:
    // index(n, m) = n*(n+1)/2 + m, for 0 <= m <= n.
    const GravityCoefficientData coefficients =
        GravityCoefficientReader::read(coefficient_file_, degree_);

    // If the file contains metadata, use its normalization information
    // rather than assuming that the input coefficients are normalized.
    if (coefficients.has_metadata &&
        coefficients.normalization != GravityNormalization::FullyNormalized) {
        throw std::invalid_argument(
            "PinesGravity requires fully normalized gravity coefficients");
    }

    // Use the coefficient-file gravitational constants when metadata
    // is available. Otherwise, retain the constructor arguments.
    if (coefficients.has_metadata) {
        if (!std::isfinite(coefficients.mu) || coefficients.mu <= 0.0 ||
            !std::isfinite(coefficients.reference_radius) ||
            coefficients.reference_radius <= 0.0) {
            throw std::invalid_argument(
                "Gravity coefficient metadata contains invalid mu or reference radius");
        }

        mu_ = coefficients.mu;
        reference_radius_ = coefficients.reference_radius;
    }

    // Copy the coefficients into the Pines model.
    // The reader has already validated the coefficient records,
    // checked for duplicate entries, and applied the requested degree.
    const std::size_t coefficient_count =
        index(degree_ + 1, 0);

    if (coefficients.C.size() != coefficient_count ||
        coefficients.S.size() != coefficient_count) {
        throw std::invalid_argument(
            "GravityCoefficientReader returned coefficient arrays with an unexpected size");
    }

    c_ = coefficients.C;
    s_ = coefficients.S;

    // The fully normalized spherical-harmonic expansion uses C_00 = 1.
    // The reader also initializes this value, but explicitly verify it here.
    if (c_.empty() || c_[index(0, 0)] != 1.0) {
        throw std::invalid_argument(
            "PinesGravity requires C00 = 1.0");
    }

    // Precompute recurrence coefficients used during gravity evaluation.
    //
    // The acceleration recurrence requires Legendre-function values
    // through degree + 1, so g_ and h_ need one additional row beyond
    // the requested gravity-field degree.
    const std::size_t recurrence_count =
        index(degree_ + 2, 0);

    g_.assign(recurrence_count, 0.0);
    h_.assign(recurrence_count, 0.0);

    // alpha_ and beta_ are used in the horizontal and radial recurrence
    // contributions. They use the same triangular indexing as C_nm/S_nm.
    alpha_.assign(coefficient_count, 0.0);
    beta_.assign(coefficient_count, 0.0);

    // Diagonal and superdiagonal factors are stored separately because
    // they are reused repeatedly during evaluation.
    diagonal_.assign(
        static_cast<std::size_t>(degree_) + 2, 0.0);

    superdiagonal_.resize(
        static_cast<std::size_t>(degree_) + 1);

    // Precompute normalized Legendre recurrence factors.
    //
    // g_[index(n,m)] and h_[index(n,m)] scale the previous one or two
    // degree rows when constructing the next recurrence row.
    // See reference [1] for the underlying Pines recurrence formulation.
    for (int n = 1; n <= degree_ + 1; ++n) {
        const double nd = static_cast<double>(n);

        diagonal_[n] =
            std::sqrt((2.0 * nd + 1.0) / (2.0 * nd));

        for (int m = 0; m < n; ++m) {
            const double md = static_cast<double>(m);
            const std::size_t k = index(n, m);

            g_[k] = std::sqrt(
                (2.0 * nd + 1.0) * (2.0 * nd - 1.0) /
                ((nd - md) * (nd + md)));

            // h_ is needed for the two-row recurrence when n >= 2 and
            // the order is below the subdiagonal boundary.
            if (n >= 2 && n - m - 1 > 0) {
                h_[k] = std::sqrt(
                    (nd + md - 1.0) * (2.0 * nd + 1.0) *
                    (nd - md - 1.0) /
                    ((nd + md) * (nd - md) * (2.0 * nd - 3.0)));
            }
        }
    }

    // Precompute alpha and beta factors for each coefficient.
    // These factors reduce repeated square-root calculations during
    // acceleration evaluations.
    for (int n = 0; n <= degree_; ++n) {
        const double nd = static_cast<double>(n);

        superdiagonal_[n] =
            std::sqrt(2.0 * nd + 3.0);

        for (int m = 0; m <= n; ++m) {
            const double md = static_cast<double>(m);
            const std::size_t k = index(n, m);

            alpha_[k] =
                std::sqrt((nd - md) * (nd + md + 1.0));

            beta_[k] = std::sqrt(
                (2.0 * nd + 1.0) *
                (nd + md + 2.0) *
                (nd + md + 1.0) /
                (2.0 * nd + 3.0));

            // The order-zero terms use a sqrt(2) normalization adjustment
            // consistent with the recurrence convention used by the model.
            if (m == 0) {
                alpha_[k] /= sqrt_two;
                beta_[k] /= sqrt_two;
            }
        }
    }
}

// Evaluate gravitational acceleration at a Cartesian position.
//
// The input position and returned acceleration use the unit system
// associated with mu_ and reference_radius_. For example, if distances
// are in kilometres and mu_ is in km^3/s^2, acceleration is in km/s^2.
std::array<double, 3> PinesGravityModel::acceleration(
    const std::array<double, 3>& position) const {

    // Reject non-finite coordinates before computing the radius.
    for (double coordinate : position) {
        if (!std::isfinite(coordinate)) {
            throw std::invalid_argument(
                "Pines position must be finite");
        }
    }

    const double x = position[0];
    const double y = position[1];
    const double z = position[2];

    // Cartesian distance from the gravity-field origin.
    const double radius = std::sqrt(x * x + y * y + z * z);

    // The gravity evaluation is undefined at the origin. Also reject
    // overflow in the radius calculation.
    if (!(radius > 0.0) || !std::isfinite(radius)) {
        throw std::invalid_argument(
            "Pines position must have a finite, nonzero radius");
    }

    // Evaluate the Pines spherical-harmonic acceleration recurrence.
    const auto result = acceleration_scalar<double>(position);

    // Guard against overflow or non-finite results from extreme inputs
    // or unsuitable model parameters.
    for (double component : result) {
        if (!std::isfinite(component)) {
            throw std::overflow_error(
                "Pines acceleration overflow; check radius and model constants");
        }
    }

    return result;
}

}  // namespace ASSET
