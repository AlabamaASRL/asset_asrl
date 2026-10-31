
#include "GravityCoefficientReader.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ASSET {
namespace {

// Determine whether a line contains only whitespace.
bool is_blank(const std::string& line) {

    return line.find_first_not_of(" \t\r\n") ==
           std::string::npos;
}


// Normalize a record for stream parsing.
//
// Commas become whitespace, allowing the same parser to read
// comma-separated GRAIL records and whitespace-delimited records.
// Fortran D exponents are converted to standard E exponents.
std::string normalize_record(std::string line) {

    const std::size_t comment = line.find_first_of("#!");

    if (comment != std::string::npos) {
        line.erase(comment);
    }

    std::replace(line.begin(), line.end(), ',', ' ');
    std::replace(line.begin(), line.end(), 'D', 'E');
    std::replace(line.begin(), line.end(), 'd', 'e');

    return line;
}


// Report malformed records with their original line number.
[[noreturn]] void invalid_line(
    int line_number,
    const std::string& reason) {

    throw std::invalid_argument(
        "Invalid gravity coefficient record at line " +
        std::to_string(line_number) + ": " + reason);
}


// Parse the GRAIL SHADR metadata header.
//
// Fields:
//
//     0: Reference radius (km)
//     1: GM (km^3/s^2)
//     2: GM uncertainty
//     3: Maximum degree
//     4: Maximum order
//     5: Normalization state
//     6: Reference longitude
//     7: Reference latitude
//
void parse_grail_header(
    const std::string& line,
    GravityCoefficientData& data) {

    std::istringstream row(normalize_record(line));
    row.imbue(std::locale::classic());

    int normalization = -1;

    if (!(row >>
          data.reference_radius >>
          data.mu >>
          data.mu_uncertainty >>
          data.degree >>
          data.order >>
          normalization >>
          data.reference_longitude >>
          data.reference_latitude)) {

        throw std::invalid_argument(
            "Invalid GRAIL SHADR metadata header");
    }

    std::string extra;

    if (row >> extra) {
        throw std::invalid_argument(
            "Unexpected extra fields in GRAIL SHADR metadata header");
    }

    if (!std::isfinite(data.reference_radius) ||
        data.reference_radius <= 0.0 ||
        !std::isfinite(data.mu) ||
        data.mu <= 0.0 ||
        !std::isfinite(data.mu_uncertainty) ||
        !std::isfinite(data.reference_longitude) ||
        !std::isfinite(data.reference_latitude)) {

        throw std::invalid_argument(
            "GRAIL metadata contains invalid numerical values");
    }

    if (data.degree < 0 ||
        data.order < 0 ||
        data.order > data.degree) {

        throw std::invalid_argument(
            "Invalid GRAIL maximum degree or order");
    }

    switch (normalization) {

        case 0:
            data.normalization =
                GravityNormalization::Unnormalized;
            break;

        case 1:
            data.normalization =
                GravityNormalization::FullyNormalized;
            break;

        case 2:
            data.normalization =
                GravityNormalization::Other;
            break;

        default:
            throw std::invalid_argument(
                "Unknown GRAIL normalization state");
    }

    data.has_metadata = true;
    data.format = "GRAIL_SHADR";
}


// Parse a coefficient record.
//
// GRAIL SHADR:
//
//     n, m, C_nm, S_nm, sigma_Cnm, sigma_Snm
//
// Whitespace-delimited:
//
//     n m C_nm S_nm
//
void parse_coefficient_record(
    const std::string& line,
    int line_number,
    int requested_degree,
    int maximum_degree,
    int maximum_order,
    bool grail_format,
    GravityCoefficientData& data,
    std::vector<bool>& seen) {

    std::istringstream row(normalize_record(line));
    row.imbue(std::locale::classic());

    int n = -1;
    int m = -1;

    double c = 0.0;
    double s = 0.0;

    double sigma_c = 0.0;
    double sigma_s = 0.0;

    if (!(row >> n >> m >> c >> s)) {
        invalid_line(
            line_number,
            "expected degree, order, C_nm, and S_nm");
    }

    if (grail_format) {

        if (!(row >> sigma_c >> sigma_s)) {
            invalid_line(
                line_number,
                "GRAIL records require six fields");
        }

        if (!std::isfinite(sigma_c) ||
            !std::isfinite(sigma_s)) {

            invalid_line(
                line_number,
                "coefficient uncertainties must be finite");
        }
    }

    std::string extra;

    if (row >> extra) {
        invalid_line(
            line_number,
            "unexpected extra coefficient fields");
    }

    if (n < 0 || m < 0 || m > n) {
        invalid_line(
            line_number,
            "require degree >= 0 and 0 <= order <= degree");
    }

    if (n > maximum_degree || m > maximum_order) {
        invalid_line(
            line_number,
            "coefficient exceeds declared maximum degree or order");
    }

    if (!std::isfinite(c) || !std::isfinite(s)) {
        invalid_line(
            line_number,
            "coefficients must be finite");
    }

    // Validate records beyond the requested truncation degree,
    // but do not store them.
    if (n > requested_degree) {
        return;
    }

    const std::size_t k =
        GravityCoefficientData::index(n, m);

    if (seen[k]) {
        invalid_line(
            line_number,
            "duplicate degree/order coefficient");
    }

    seen[k] = true;

    data.C[k] = c;
    data.S[k] = s;
}


// Find the degree associated with a triangular storage index.
int degree_from_index(std::size_t k) {

    int n = 0;

    while (GravityCoefficientData::index(n + 1, 0) <= k) {
        ++n;
    }

    return n;
}

}  // namespace


GravityCoefficientData GravityCoefficientReader::read(
    const std::string& filename,
    int requested_degree) {

    if (requested_degree < 0) {
        throw std::invalid_argument(
            "Requested gravity degree cannot be negative");
    }

    std::ifstream input(filename);

    if (!input) {
        throw std::invalid_argument(
            "Cannot open gravity coefficient file: " + filename);
    }

    input.imbue(std::locale::classic());

    GravityCoefficientData data;

    data.source_file = filename;

    const std::size_t coefficient_count =
        GravityCoefficientData::index(
            requested_degree + 1, 0);

    data.C.assign(coefficient_count, 0.0);
    data.S.assign(coefficient_count, 0.0);

    // Fully normalized spherical harmonics use C_00 = 1.
    // The reader does not infer normalization for other files.
    data.C[GravityCoefficientData::index(0, 0)] = 1.0;

    std::vector<bool> seen(coefficient_count, false);

    // Locate the first nonblank record.
    std::string line;
    std::string first_line;

    int line_number = 0;
    int first_line_number = 0;

    while (std::getline(input, line)) {

        ++line_number;

        if (is_blank(normalize_record(line))) {
            continue;
        }

        first_line = line;
        first_line_number = line_number;

        break;
    }

    if (first_line.empty()) {
        throw std::invalid_argument(
            "Gravity coefficient file is empty: " + filename);
    }

    // GRAIL SHADR begins with a comma-separated metadata header.
    const bool grail_format =
        first_line.find(',') != std::string::npos;

    int maximum_degree = requested_degree;
    int maximum_order = requested_degree;

    if (grail_format) {

        parse_grail_header(first_line, data);

        maximum_degree = data.degree;
        maximum_order = data.order;

        if (requested_degree > maximum_degree) {
            throw std::invalid_argument(
                "Requested degree exceeds the model maximum degree");
        }

        if (requested_degree > maximum_order) {
            throw std::invalid_argument(
                "Requested degree exceeds the model maximum order");
        }

    } else {

        // The reader cannot infer GM, reference radius, or
        // normalization from a generic coefficient file.
        data.format = "WHITESPACE";
        data.degree = requested_degree;
        data.order = requested_degree;

        // The maximum available degree is checked after parsing.
        maximum_degree = 100000;
        maximum_order = 100000;

        parse_coefficient_record(
            first_line,
            first_line_number,
            requested_degree,
            maximum_degree,
            maximum_order,
            false,
            data,
            seen);
    }

    // Parse the remaining coefficient records.
    while (std::getline(input, line)) {

        ++line_number;

        if (is_blank(normalize_record(line))) {
            continue;
        }

        parse_coefficient_record(
            line,
            line_number,
            requested_degree,
            maximum_degree,
            maximum_order,
            grail_format,
            data,
            seen);
    }

    if (input.bad()) {
        throw std::runtime_error(
            "Error reading gravity coefficient file: " + filename);
    }

    // Ensure that the file actually contains records reaching
    // the requested degree.
    int maximum_loaded_degree = -1;

    for (std::size_t k = 0; k < seen.size(); ++k) {

        if (seen[k]) {
            maximum_loaded_degree =
                std::max(
                    maximum_loaded_degree,
                    degree_from_index(k));
        }
    }

    if (maximum_loaded_degree < requested_degree) {
        throw std::invalid_argument(
            "Gravity coefficient file does not reach the requested degree");
    }

    return data;
}

}  // namespace ASSET
