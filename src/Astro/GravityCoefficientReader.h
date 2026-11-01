
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace ASSET {

// Identifies the normalization convention of the coefficient data.
enum class GravityNormalization {
    Unknown,
    Unnormalized,
    FullyNormalized,
    Other
};


// Generic spherical-harmonic gravity coefficient data.
//
// Coefficients use triangular storage:
//
//     index(n,m) = n*(n+1)/2 + m
//
// The arrays contain C_nm and S_nm through the requested degree.
struct GravityCoefficientData {

    int degree = 0;
    int order = 0;

    double mu = 0.0;
    double reference_radius = 0.0;
    double mu_uncertainty = 0.0;

    double reference_longitude = 0.0;
    double reference_latitude = 0.0;

    bool has_metadata = false;

    GravityNormalization normalization =
        GravityNormalization::Unknown;

    std::string source_file;
    std::string format;

    std::vector<double> C;
    std::vector<double> S;

    static std::size_t index(int n, int m) {
        return static_cast<std::size_t>(n) *
               static_cast<std::size_t>(n + 1) / 2 +
               static_cast<std::size_t>(m);
    }
};


// Generic spherical-harmonic coefficient file reader.
//
// Supported formats:
//
//     1. GRAIL SHADR ASCII
//        - Comma-separated metadata header.
//        - Six-column coefficient records:
//          n, m, C_nm, S_nm, sigma_Cnm, sigma_Snm.
//
//     2. Whitespace-delimited coefficient files
//        - Four-column records:
//          n, m, C_nm, S_nm.
//        - Supports Fortran D/d exponents.
//
// The reader does not perform gravity evaluation or normalization
// conversion. It returns the coefficient values as stored in the file.
class GravityCoefficientReader {

public:

    static GravityCoefficientData read(
        const std::string& filename,
        int requested_degree);

};

}  // namespace ASSET
