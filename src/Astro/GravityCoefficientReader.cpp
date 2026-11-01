
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
    return line.find_first_not_of(" \t\r\n") == std::string::npos;
}


// Normalize a record for stream parsing.
//
// Commas become whitespace, allowing the same parser to read
// comma-separated GRAIL records and whitespace-delimited EGM96 records.
//
// Fortran D exponents are converted to standard E exponents.
// Comments beginning with # or ! are removed.
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


// Parse an integer field using the classic locale.
int parse_integer(
    const std::string& value,
    int line_number,
    const std::string& field_name) {

    std::istringstream parser(value);
    parser.imbue(std::locale::classic());

    int result = 0;
    char extra = '\0';

    if (!(parser >> result) || (parser >> extra)) {
        invalid_line(
            line_number,
            "invalid integer field: " + field_name);
    }

    return result;
}


// Parse a floating-point field using the classic locale.
double parse_double(
    const std::string& value,
    int line_number,
    const std::string& field_name) {

    std::istringstream parser(value);
    parser.imbue(std::locale::classic());

    double result = 0.0;
    char extra = '\0';

    if (!(parser >> result) || (parser >> extra)) {
        invalid_line(
            line_number,
            "invalid floating-point field: " + field_name);
    }

    if (!std::isfinite(result)) {
        invalid_line(
            line_number,
            "non-finite value in field: " + field_name);
    }

    return result;
}


// Split a normalized record into whitespace-delimited fields.
std::vector<std::string> split_fields(const std::string& line) {

    std::istringstream row(line);
    row.imbue(std::locale::classic());

    std::vector<std::string> fields;
    std::string field;

    while (row >> field) {
        fields.push_back(field);
    }

    return fields;
}


// Parse the GRAIL SHADR metadata header.
//
// Fields:
//   0: Reference radius (km)
//   1: GM (km^3/s^2)
//   2: GM uncertainty
//   3: Maximum degree
//   4: Maximum order
//   5: Normalization state
//   6: Reference longitude
//   7: Reference latitude
void parse_grail_header(
    const std::string& line,
    GravityCoefficientData& data) {

    const std::vector<std::string> fields =
        split_fields(normalize_record(line));

    if (fields.size() != 8) {
        throw std::invalid_argument(
            "Invalid GRAIL SHADR metadata header: expected eight fields");
    }

    data.reference_radius =
        parse_double(fields[0], 1, "reference radius");

    data.mu =
        parse_double(fields[1], 1, "GM");

    data.mu_uncertainty =
        parse_double(fields[2], 1, "GM uncertainty");

    data.degree =
        parse_integer(fields[3], 1, "maximum degree");

    data.order =
        parse_integer(fields[4], 1, "maximum order");

    const int normalization =
        parse_integer(fields[5], 1, "normalization");

    data.reference_longitude =
        parse_double(fields[6], 1, "reference longitude");

    data.reference_latitude =
        parse_double(fields[7], 1, "reference latitude");

    if (data.reference_radius <= 0.0 || data.mu <= 0.0) {
        throw std::invalid_argument(
            "GRAIL metadata requires positive reference radius and GM");
    }

    if (data.mu_uncertainty < 0.0) {
        throw std::invalid_argument(
            "GRAIL GM uncertainty cannot be negative");
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
//   n, m, C_nm, S_nm, sigma_Cnm, sigma_Snm
//
// EGM96 / whitespace-delimited:
//   n m C_nm S_nm
//
// or:
//
//   n m C_nm S_nm sigma_Cnm sigma_Snm
//
// Uncertainty fields are validated but not stored.
void parse_coefficient_record(
    const std::string& line,
    int line_number,
    int requested_degree,
    int maximum_degree,
    int maximum_order,
    bool grail_format,
    GravityCoefficientData& data,
    std::vector<bool>& seen,
    int& maximum_loaded_degree) {

    const std::vector<std::string> fields =
        split_fields(normalize_record(line));

    // GRAIL requires six fields.
    // EGM96 accepts either four or six fields.
    if (grail_format) {

        if (fields.size() != 6) {
            invalid_line(
                line_number,
                "GRAIL records require exactly six fields");
        }

    } else {

        if (fields.size() != 4 && fields.size() != 6) {
            invalid_line(
                line_number,
                "expected four or six fields: n, m, C_nm, S_nm, and optional uncertainties");
        }
    }

    const int n =
        parse_integer(fields[0], line_number, "degree");

    const int m =
        parse_integer(fields[1], line_number, "order");

    const double c =
        parse_double(fields[2], line_number, "C_nm");

    const double s =
        parse_double(fields[3], line_number, "S_nm");

    // Validate optional coefficient uncertainties.
    // These values are not required for gravity evaluation.
    if (fields.size() == 6) {

        const double sigma_c =
            parse_double(fields[4], line_number, "sigma_Cnm");

        const double sigma_s =
            parse_double(fields[5], line_number, "sigma_Snm");

        if (sigma_c < 0.0 || sigma_s < 0.0) {
            invalid_line(
                line_number,
                "coefficient uncertainties cannot be negative");
        }
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

    maximum_loaded_degree =
        std::max(maximum_loaded_degree, n);

    // Validate records beyond the requested truncation degree,
    // but do not store them.
    if (n > requested_degree) {
        return;
    }

    const std::size_t k =
        GravityCoefficientData::index(n, m);

    if (k >= data.C.size()) {
        invalid_line(
            line_number,
            "coefficient index exceeds allocated storage");
    }

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


// =============================================================================
// GRAVITY COEFFICIENT READER
// =============================================================================

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
        GravityCoefficientData::index(requested_degree + 1, 0);

    data.C.assign(coefficient_count, 0.0);
    data.S.assign(coefficient_count, 0.0);

    // Initialize C_00.
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
    // EGM96 coefficient files begin directly with whitespace-separated
    // degree/order records.
    const bool grail_format =
        first_line.find(',') != std::string::npos;

    int maximum_degree = requested_degree;
    int maximum_order = requested_degree;

    int maximum_loaded_degree = -1;

    if (grail_format) {

        parse_grail_header(first_line, data);

        maximum_degree = data.degree;
        maximum_order = data.order;

        if (requested_degree > maximum_degree) {
            throw std::invalid_argument(
                "Requested degree exceeds the GRAIL model maximum degree");
        }

        if (requested_degree > maximum_order) {
            throw std::invalid_argument(
                "Requested degree exceeds the GRAIL model maximum order");
        }

    } else {

        // EGM96 does not provide metadata in the coefficient records.
        // GM, reference radius, and normalization must be supplied
        // separately by the Pines model.
        data.format = "EGM96_WHITESPACE";
        data.degree = requested_degree;
        data.order = requested_degree;

        // No maximum degree/order is declared in this file format.
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
            seen,
            maximum_loaded_degree);
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
            seen,
            maximum_loaded_degree);
    }

    if (input.bad()) {
        throw std::runtime_error(
            "Error reading gravity coefficient file: " + filename);
    }

    // EGM96 commonly starts at degree 2 because C_00 is initialized
    // internally and degree-1 terms are omitted.
    //
    // Therefore, requests for degree 0 or 1 do not require explicit
    // coefficient records. For degree 2 and above, verify that the file
    // contains records reaching the requested degree.
    if (requested_degree >= 2 &&
        maximum_loaded_degree < requested_degree) {

        throw std::invalid_argument(
            "Gravity coefficient file does not reach the requested degree");
    }

    // For GRAIL, the header declares the available maximum degree/order.
    // For EGM96, the maximum loaded degree is determined from the records.
    return data;
}

}  // namespace ASSET
