#pragma once

#include "CommonFunctions/CommonFunctions.h"
#include "VectorFunctionTypeErasure/GenericFunction.h"

namespace ASSET {

  /////////////////////// Scalar Multiplication and
  /// Division//////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////

  template<class Derived, int IR, int OR>
  decltype(auto) operator*(const DenseFunctionBase<Derived, IR, OR>& func, double s) {
    return Scaled<Derived>(func.derived(), s);
  }

  template<int IR, int OR>
  auto operator*(const GenericFunction<IR, OR>& func, double s) {
    return Scaled<GenericFunction<IR, OR>>(func, s);
  }

  template<class Derived, int IR, int OR, class VALUE>
  decltype(auto) operator*(const DenseFunctionBase<Derived, IR, OR>& func, StaticScaleBase<VALUE> s) {
    return StaticScaled<Derived, VALUE>(func.derived());
  }

  template<class Derived, int IR, int OR>
  decltype(auto) operator*(double s, const DenseFunctionBase<Derived, IR, OR>& func) {
    return Scaled<Derived>(func.derived(), s);
  }

  template<int IR, int OR>
  auto operator*(double s, const GenericFunction<IR, OR>& func) {
    return Scaled<GenericFunction<IR, OR>>(func, s);
  }

  template<class Derived, int IR, int OR, class OutType>
  decltype(auto) operator*(const DenseFunctionBase<Derived, IR, OR>& func,
                           const Eigen::MatrixBase<OutType>& s) {
    return RowScaled<Derived>(func.derived(), s);
  }
  template<class Derived, int IR, int OR, class OutType>
  decltype(auto) operator*(const Eigen::MatrixBase<OutType>& s,
                           const DenseFunctionBase<Derived, IR, OR>& func) {
    return RowScaled<Derived>(func.derived(), s);
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////
  template<class Derived>
  decltype(auto) operator*(const Scaled<Derived>& func, double s) {
    return Scaled<Derived>(func.Scaled_func, func.Scale_value * s);
  }
  template<class Derived>
  decltype(auto) operator*(double s, const Scaled<Derived>& func) {
    return Scaled<Derived>(func.Scaled_func, func.Scale_value * s);
  }
  template<class Derived>
  decltype(auto) operator*(const RowScaled<Derived>& func, double s) {
    return RowScaled<Derived>(func.RowScaled_func, func.RowScale_values * s);
  }
  template<class Derived>
  decltype(auto) operator*(double s, const RowScaled<Derived>& func) {
    return RowScaled<Derived>(func.RowScaled_func, func.RowScale_values * s);
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////
  template<class Derived, class OutType>
  decltype(auto) operator*(const RowScaled<Derived>& func, const Eigen::MatrixBase<OutType>& s) {
    return RowScaled<Derived>(func.RowScaled_func, func.RowScale_values.cwiseProduct(s));
  }
  template<class Derived, class OutType>
  decltype(auto) operator*(const Eigen::MatrixBase<OutType>& s, const RowScaled<Derived>& func) {
    return RowScaled<Derived>(func.RowScaled_func, func.RowScale_values.cwiseProduct(s));
  }
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////

  template<class Derived, int IR, int OR>
  decltype(auto) operator/(const DenseFunctionBase<Derived, IR, OR>& func, double s) {
    return Scaled<Derived>(func.derived(), 1.0 / s);
  }
  template<class Derived, int IR, int OR, class OutType>
  decltype(auto) operator/(const DenseFunctionBase<Derived, IR, OR>& func,
                           const Eigen::MatrixBase<OutType>& s) {
    return RowScaled<Derived>(func.derived(), s.cwiseInverse());
  }
  template<class Derived>
  decltype(auto) operator/(const Scaled<Derived>& func, double s) {
    return Scaled<Derived>(func.func, func.Scale_value / s);
  }

  template<class Derived, int IR>
  decltype(auto) operator/(double s, const DenseFunctionBase<Derived, IR, 1>& func) {
    return Scaled<CwiseInverse<Derived>>(CwiseInverse<Derived>(func.derived()), s);
  }

  /////////////////////// Scalar Addition
  /// Subtraction///////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  template<class Derived, int IR, int OR, class OutType>
  decltype(auto) operator+(const DenseFunctionBase<Derived, IR, OR>& func,
                           const Eigen::MatrixBase<OutType>& s) {
    return FunctionPlusVector<Derived>(func.derived(), s);
  }

  template<int IR, int OR, class OutType>
  auto operator+(const GenericFunction<IR, OR>& func, const Eigen::MatrixBase<OutType>& s) {
    return FunctionPlusVector<GenericFunction<IR, OR>>(func, s);
  }

  template<class Derived, int IR, int OR, class OutType>
  decltype(auto) operator+(const Eigen::MatrixBase<OutType>& s,
                           const DenseFunctionBase<Derived, IR, OR>& func) {
    return FunctionPlusVector<Derived>(func.derived(), s);
  }

  template<int IR, int OR, class OutType>
  auto operator+(const Eigen::MatrixBase<OutType>& s, const GenericFunction<IR, OR>& func) {
    return FunctionPlusVector<GenericFunction<IR, OR>>(func, s);
  }

  template<class Derived, int IR, int OR, class OutType>
  decltype(auto) operator-(const DenseFunctionBase<Derived, IR, OR>& func,
                           const Eigen::MatrixBase<OutType>& s) {
    return FunctionPlusVector<Derived>(func.derived(), (-s));
  }

  template<int IR, int OR, class OutType>
  auto operator-(const GenericFunction<IR, OR>& func, const Eigen::MatrixBase<OutType>& s) {
    return FunctionPlusVector<GenericFunction<IR, OR>>(func, (-s));
  }

  template<class Derived, int IR, int OR, class OutType>
  decltype(auto) operator-(const Eigen::MatrixBase<OutType>& s,
                           const DenseFunctionBase<Derived, IR, OR>& func) {
    return FunctionPlusVector<Scaled<Derived>>(Scaled<Derived>(func.derived(), -1.0), s);
  }

  template<class Derived, int IR>
  decltype(auto) operator+(const DenseFunctionBase<Derived, IR, 1>& func, double s) {
    Vector1<double> st;
    st[0] = s;
    return FunctionPlusVector<Derived>(func.derived(), st);
  }
  template<class Derived, int IR>
  decltype(auto) operator+(double s, const DenseFunctionBase<Derived, IR, 1>& func) {
    Vector1<double> st;
    st[0] = s;
    return FunctionPlusVector<Derived>(func.derived(), st);
  }
  template<class Derived, int IR>
  decltype(auto) operator-(const DenseFunctionBase<Derived, IR, 1>& func, double s) {
    Vector1<double> st;
    st[0] = s;
    return FunctionPlusVector<Derived>(func.derived(), (-st));
  }
  template<class Derived, int IR>
  decltype(auto) operator-(double s, const DenseFunctionBase<Derived, IR, 1>& func) {
    Vector1<double> st;
    st[0] = s;

    return FunctionPlusVector<Scaled<Derived>>(Scaled<Derived>(func.derived(), -1.0), st);
  }

  /////////////////////// Function Scalar Multiplication And Division
  ///////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  template<class VFunc, int IR, int OR, class SFunc>
  decltype(auto) operator*(const DenseFunctionBase<VFunc, IR, OR>& vf,
                           const DenseFunctionBase<SFunc, IR, 1>& sf) {
    return VectorScalarFunctionProduct<VFunc, SFunc>(vf.derived(), sf.derived());
  }
  template<class VFunc, int IR, int OR, class SFunc>
  decltype(auto) operator*(const DenseFunctionBase<SFunc, IR, 1>& sf,
                           const DenseFunctionBase<VFunc, IR, OR>& vf) {
    return VectorScalarFunctionProduct<VFunc, SFunc>(vf.derived(), sf.derived());
  }
  template<class VFunc, int IR, class SFunc>
  decltype(auto) operator*(const DenseFunctionBase<SFunc, IR, 1>& sf,
                           const DenseFunctionBase<VFunc, IR, 1>& vf) {
    return VectorScalarFunctionProduct<VFunc, SFunc>(vf.derived(), sf.derived());
  }
  template<class VFunc, int IR, int OR, class SFunc>
  decltype(auto) operator/(const DenseFunctionBase<VFunc, IR, OR>& vf,
                           const DenseFunctionBase<SFunc, IR, 1>& sf) {
    return VectorScalarFunctionDivision<VFunc, SFunc>(vf.derived(), sf.derived());
  }

  /////////////////////// Function Addition
  /// Subtraction/////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  template<class Func1, int IR, int OR, class Func2>
  decltype(auto) operator+(const DenseFunctionBase<Func1, IR, OR>& f1,
                           const DenseFunctionBase<Func2, IR, OR>& f2) {
    return TwoFunctionSum<Func1, Func2>(f1.derived(), f2.derived());
  }

  template<class Func1, int IR, int OR, class Func2, class Func3>
  decltype(auto) operator+(const DenseFunctionBase<Func3, IR, OR>& f2,
                           const TwoFunctionSum<Func1, Func2>& f1) {
    return MultiFunctionSum<Func1, Func2, Func3>(f1.func1, f1.func2, f2.derived());
  }
  template<class Func1, int IR, int OR, class Func2, class Func3, class Func4>
  decltype(auto) operator+(const DenseFunctionBase<Func4, IR, OR>& f2,
                           const MultiFunctionSum<Func1, Func2, Func3>& f1) {
    return MultiFunctionSum<Func1, Func2, Func3, Func4>(
        f1.func1, f1.func2, std::get<0>(f1.funcs), f2.derived());
  }

  template<class Func1, int IR, int OR, class Func2>
  decltype(auto) operator-(const DenseFunctionBase<Func1, IR, OR>& f1,
                           const DenseFunctionBase<Func2, IR, OR>& f2) {
    return FunctionDifference<Func1, Func2>(f1.derived(), f2.derived());
  }


  ////////////Comparisons///////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////

  template<class Func1, int IR, class Func2>
  auto operator<(const DenseFunctionBase<Func1, IR, 1>& lhs, const DenseFunctionBase<Func2, IR, 1>& rhs) {
    return ConditionalStatement<Func1, Func2>(lhs.derived(), ConditionalFlags::LessThanFlag, rhs.derived());
  }
  template<class Func1, int IR, class Func2>
  auto operator<=(const DenseFunctionBase<Func1, IR, 1>& lhs, const DenseFunctionBase<Func2, IR, 1>& rhs) {
    return ConditionalStatement<Func1, Func2>(
        lhs.derived(), ConditionalFlags::LessThanEqualToFlag, rhs.derived());
  }
  template<class Func1, int IR, class Func2>
  auto operator>(const DenseFunctionBase<Func1, IR, 1>& lhs, const DenseFunctionBase<Func2, IR, 1>& rhs) {
    return ConditionalStatement<Func1, Func2>(
        lhs.derived(), ConditionalFlags::GreaterThanFlag, rhs.derived());
  }
  template<class Func1, int IR, class Func2>
  auto operator>=(const DenseFunctionBase<Func1, IR, 1>& lhs, const DenseFunctionBase<Func2, IR, 1>& rhs) {
    return ConditionalStatement<Func1, Func2>(
        lhs.derived(), ConditionalFlags::GreaterThanEqualToFlag, rhs.derived());
  }
  template<class Func1, int IR, class Func2>
  auto operator==(const DenseFunctionBase<Func1, IR, 1>& lhs, const DenseFunctionBase<Func2, IR, 1>& rhs) {
    return ConditionalStatement<Func1, Func2>(lhs.derived(), ConditionalFlags::EqualToFlag, rhs.derived());
  }


  template<class Func1, int IR>
  auto operator<(const DenseFunctionBase<Func1, IR, 1>& lhs, double rhsv) {
    Vector1<double> tmp;
    tmp[0] = rhsv;
    Constant<IR, 1> rhs(lhs.IRows(), tmp);
    return ConditionalStatement<Func1, Constant<IR, 1>>(
        lhs.derived(), ConditionalFlags::LessThanFlag, rhs.derived());
  }
  template<class Func1, int IR>
  auto operator<=(const DenseFunctionBase<Func1, IR, 1>& lhs, double rhsv) {
    Vector1<double> tmp;
    tmp[0] = rhsv;
    Constant<IR, 1> rhs(lhs.IRows(), tmp);
    return ConditionalStatement<Func1, Constant<IR, 1>>(
        lhs.derived(), ConditionalFlags::LessThanEqualToFlag, rhs.derived());
  }
  template<class Func1, int IR>
  auto operator>(const DenseFunctionBase<Func1, IR, 1>& lhs, double rhsv) {
    Vector1<double> tmp;
    tmp[0] = rhsv;
    Constant<IR, 1> rhs(lhs.IRows(), tmp);
    return ConditionalStatement<Func1, Constant<IR, 1>>(
        lhs.derived(), ConditionalFlags::GreaterThanFlag, rhs.derived());
  }
  template<class Func1, int IR>
  auto operator>=(const DenseFunctionBase<Func1, IR, 1>& lhs, double rhsv) {
    Vector1<double> tmp;
    tmp[0] = rhsv;
    Constant<IR, 1> rhs(lhs.IRows(), tmp);
    return ConditionalStatement<Func1, Constant<IR, 1>>(
        lhs.derived(), ConditionalFlags::GreaterThanEqualToFlag, rhs.derived());
  }
  template<class Func1, int IR>
  auto operator==(const DenseFunctionBase<Func1, IR, 1>& lhs, double rhsv) {
    Vector1<double> tmp;
    tmp[0] = rhsv;
    Constant<IR, 1> rhs(lhs.IRows(), tmp);
    return ConditionalStatement<Func1, Constant<IR, 1>>(
        lhs.derived(), ConditionalFlags::EqualToFlag, rhs.derived());
  }

  template<class Func1, int IR>
  auto operator<(double lhs, const DenseFunctionBase<Func1, IR, 1>& rhs) {
    return (rhs > lhs);
  }
  template<class Func1, int IR>
  auto operator<=(double lhs, const DenseFunctionBase<Func1, IR, 1>& rhs) {
    return (rhs >= lhs);
  }
  template<class Func1, int IR>
  auto operator>(double lhs, const DenseFunctionBase<Func1, IR, 1>& rhs) {
    return (rhs < lhs);
  }
  template<class Func1, int IR>
  auto operator>=(double lhs, const DenseFunctionBase<Func1, IR, 1>& rhs) {
    return (rhs <= lhs);
  }
  template<class Func1, int IR>
  auto operator==(double lhs, const DenseFunctionBase<Func1, IR, 1>& rhs) {
    return (rhs == lhs);
  }


  ////////////////////// Matrix Products /////////////////////////////////

  template<class M1, class M2, int M1Rows, int M1Cols_M2Rows, int M2Cols, int M1Major, int M2Major>
  decltype(auto) operator*(const MatrixFunctionView<M1, M1Rows, M1Cols_M2Rows, M1Major>& m1,
                           const MatrixFunctionView<M2, M1Cols_M2Rows, M2Cols, M2Major>& m2) {

    using MatFunc1 = MatrixFunctionView<M1, M1Rows, M1Cols_M2Rows, M1Major>;
    using MatFunc2 = MatrixFunctionView<M2, M1Cols_M2Rows, M2Cols, M2Major>;

    return MatrixFunctionProduct<MatFunc1, MatFunc2>(m1, m2);
  }
  ///////////////////////////////////////////////////////////////////////

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace ASSET
