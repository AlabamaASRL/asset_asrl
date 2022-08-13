#pragma once

namespace ASSET {
namespace doc {

const char* const GenericFunction_IRows =
    "Get the input size of this function\n\n"
    ":returns: int: Number of rows for input vector";

const char* const GenericFunction_ORows =
    "Get the output size of this function\n\n"
    ":returns: int: Number of rows for output vector";

const char* const GenericFunction_name =
    "Returns a string of the C++ type name.\n\n"
    ":returns: C++ type name";

const char* const GenericFunction_compute =
    "'Compute' the output of the function for the given input\n\n"
    ":param arg0: Numpy array of input variables (Size = IRows())\n"
    ":returns: Numpy array of output variables (Size = ORows())";

const char* const GenericFunction_jacobian =
    "Calculate the jacobian matrix at the given input variables\n\n"
    ":param arg0: Numpy array of input variables (Size = IRows())\n"
    ":returns: Numpy matrix of first derivatives  (Size = ORows() X IRows())";

const char* const GenericFunction_adjointgradient =
    "Calculate the adjoint gradient at the input 'arg0' with respect to the "
    "variables 'arg1'. The adjoint gradient is the Jacobian matrix multiplied "
    "on the left by a vector of size ORows\n\n"
    ":param arg0: Numpy array of input variables (Size = IRows())\n"
    ":param arg1: Numpy array of adjoint variables (Size = ORows())\n"
    ":returns: Numpy array of adjoint gradient = arg1.T*J (Size = IRows())";

const char* const GenericFunction_adjointhessian =
    "Calculate the adjoint hessian at the input 'arg0' with respect to the "
    "variables 'arg1'. The adjoint hessian is the derivative of the adjoint "
    "gradient.\n\n"
    ":param arg0: Numpy array of input variables (Size = IRows())\n"
    ":param arg1: Numpy array of adjoint variables (Size = ORows())\n"
    ":returns: Numpy matrix of adjoint hessian (Size = IRows() X IRows())";

const char* const GenericFunction_input_domain = "";

const char* const GenericFunction_is_linear =
    "Boolean function stating whether the function is linear. Takes no "
    "arguments.\n\n"
    ":returns: True if linear, False otherwise";

const char* const GenericFunction_SuperTest =
    "Perform timing tests using vectorized scalars.\n\n"
    ":param arg0: Input variables at which the function is evaluated\n"
    ":param arg1: Integer number of times to run test\n"
    ":type arg1: int\n"
    ":returns: void";

const char* const GenericFunction_eval1 =
    "Create nested function from Generic Function. arg0's outputs are passed "
    "to this function's inputs.\n\n"
    ":param arg0: Inner function\n"
    ":returns: Nested function object";

const char* const GenericFunction_eval2 =
    "Create nested function from dynamic Segment type. arg0's outputs are "
    "passed to this functions's inputs.";

const char* const GenericFunction_eval3 =
    "Create nested function from Segment of size 1. arg0's outputs are passed "
    "to this function's inputs.\n\n"
    ":param arg0: Inner function\n"
    ":returns: Nested function object";

const char* const GenericFunction_eval4 =
    "Create nested function from Segment of size 2. arg0's outputs are passed "
    "to this function's inputs.\n\n"
    ":param arg0: Inner function\n"
    ":returns: Nested function object";

const char* const GenericFunction_eval5 =
    "Create nested function from Segment of size 3. arg0's outputs are passed "
    "to this function's inputs.\n\n"
    ":param arg0: Inner function\n"
    ":returns: Nested function object";

const char* const GenericFunction_eval6 =
    "Map to this function from a higher-dimensional input. Given a large input "
    "vector, evaluate this function at indices given by arg1.\n\n"
    ":param arg0: Larger vector dimension (int)\n"
    ":param arg1: Vector of indices of large vector that compose smaller "
    "vector";

const char* const GenericFunction_padded_lower =
    "Raise the output dimension by appending zeros to the end.\n\n"
    ":param arg0: Number of zeros to append\n"
    ":returns: Function with higher ouput dimension and original ouput at the "
    "top";

const char* const GenericFunction_padded_upper =
    "Raise the output dimension by appending zeros to the beginning.\n\n"
    ":param arg0: Number of zeros to append\n"
    ":returns: Function with higher output dimension and original output at "
    "the bottom";

const char* const GenericFunction_rpt =
    "Run, Print, Time. Run the function, print the output, show how long it "
    "took.\n\n"
    ":param arg0: Vector of inputs at which to evaluate the function\n"
    ":param arg1: Number of times to run the evaluation";

const char* const GenericFunction_norm =
    "Takes the L2 norm (Euclidean distance) of the function output.\n\n"
    ":returns: Scalar Function that outputs the norm of the function";

const char* const GenericFunction_squared_norm =
    "Takes the square of the L2 norm (Euclidean distance) of the function "
    "output.\n\n"
    ":returns: Scalar Function that outputs the square of the norm of the "
    "function";

const char* const GenericFunction_cubed_norm =
    "Takes the cube of the L2 norm (Euclidean distance) of the function "
    "output.\n\n"
    ":returns: Scalar Function that outputs the cube of the norm of the "
    "function";

const char* const GenericFunction_inverse_norm =
    "Takes the inverse of the L2 norm (Euclidean distance) of the function "
    "output.\n\n"
    ":returns: Scalar Function that outputs the inverse of the norm of the "
    "function";

const char* const GenericFunction_inverse_squared_norm =
    "Takes the inverse of the square of the L2 norm (Euclidean distance) of "
    "the function output.\n\n"
    ":returns: Scalar Function that outputs the inverse of the square of the "
    "norm of the function";

const char* const GenericFunction_inverse_cubed_norm =
    "Takes the inverse of the cube of the L2 norm (Euclidean distance) of "
    "the function output.\n\n"
    ":returns: Scalar Function that outputs the inverse of the cube of the "
    "norm of the function";

const char* const GenericFunction_inverse_four_norm =
    "Takes the inverse of the fourth power of the L2 norm (Euclidean distance) "
    "of the function output.\n\n"
    ":returns: Scalar Function that outputs the inverse of the fourth power of "
    "the norm of the function";

const char* const GenericFunction_normalized =
    "Turns the output of the function into a unit vector.\n\n"
    ":returns: Function whose output is a unit vector along the same direction "
    "of the original function";

const char* const GenericFunction_normalized_power2 =
    "Normalizes the function output by the squared norm.\n\n"
    ":returns: Function whose output has been divided by the square of its "
    "norm";

const char* const GenericFunction_normalized_power3 =
    "Normalizes the function output by the cubed norm.\n\n"
    ":returns: Function whose output has been divided by the cube of its norm";

const char* const GenericFunction_normalized_power4 =
    "Normalizes the function output by the norm to the fourth power.\n\n"
    ":returns: Function whose output has been divided by the fourth power of "
    "its norm";

const char* const GenericFunction_normalized_power5 =
    "Normalizes the function output by the norm to the fifth power.\n\n"
    ":returns: Function whose output has been divided by the fifth power of "
    "its norm";

const char* const GenericFunction_cross1 =
    "The cross product between this function's output and another function's "
    "output.\n\n"
    ":param arg0: The other function with output size 3\n"
    ":returns: Function whose output is the cross product of this and arg0";

const char* const GenericFunction_cross2 =
    "The cross product between this function's output and a dynamic "
    "segment.\n\n"
    ":param arg0: A dynamic segment with size 3\n"
    ":returns: Function whose output is the cross product of this and arg0";

const char* const GenericFunction_cross3 =
    "The cross product between this function's output and a fixed size-3 "
    "segment.\n\n"
    ":param arg0: A fixed segment with size 3\n"
    ":returns: Function whose output is the cross product of this and arg0";

const char* const GenericFunction_squared =
    "For a function of output size 1, square the result.\n\n"
    ":returns: Scalar function whose output is the square of the original";

const char* const GenericFunction_sqrt =
    "For a function of output size 1, take the square root of the result.\n\n"
    ":returns: Scalar function whose output is the square root of the original";

const char* const GenericFunction_exp =
    "For a function of output size 1, take the exponential of the result.\n\n"
    ":returns: Scalar function whose output is the exponential of the original";

const char* const GenericFunction_sin =
    "For a function of output size 1, compute the sine of the result.\n\n"
    ":returns: Scalar function whose output is the sine of the original";

const char* const GenericFunction_cos =
    "For a function of output size 1, compute the cosine of the result.\n\n"
    ":returns: Scalar function whose output is the cosine of the original";

const char* const GenericFunction_inverse =
    "For a function of output size 1, compute the inverse of the result.\n\n"
    ":returns: Scalar function whose output is the inverse of the original";

}  // namespace doc
}  // namespace ASSET
