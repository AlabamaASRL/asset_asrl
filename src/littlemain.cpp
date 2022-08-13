#include "pch.h"
#include "CSEelimination.h"
#include "CommonFunctions/CommonFunctions.h"
#include "FunctionRegistry.h"
#include "Memoization.h"
#include "OperatorOverloads.h"
#include "VectorFunction.h"
#include "VectorFunctionTypeErasure/GenericFunction.h"
#include "VectorFunctionTypeErasure/GenericConditional.h"
#include "Solvers/ASSET_Solvers.h"
#include "Utils/ASSET_Utils.h"
#include "Utils/MemoryManagement.h"

using namespace ASSET;
// using namespace Eigen;
using namespace rubber_types;

////////////////////////////////////////////////////////////////////////////
template <class Func1, class Func2>
struct FunctionCrossProductTest
    : VectorFunction<FunctionCrossProductTest<Func1, Func2>, SZ_MAX<Func1::IRC, Func2::IRC>::value, 3> {
    using Base =
        VectorFunction<FunctionCrossProductTest<Func1, Func2>, SZ_MAX<Func1::IRC, Func2::IRC>::value, 3>;
    using Base::compute;
    DENSE_FUNCTION_BASE_TYPES(Base);
    SUB_FUNCTION_IO_TYPES(Func1);
    SUB_FUNCTION_IO_TYPES(Func2);

    Func1 func1;
    Func2 func2;

    using INPUT_DOMAIN = CompositeDomain<Base::IRC, typename Func1::INPUT_DOMAIN,
        typename Func2::INPUT_DOMAIN>;

#if defined(_WIN32)
    static const bool IsVectorizable = Func1::IsVectorizable && Func2::IsVectorizable;
#else
    static const bool IsVectorizable = false;
#endif 

    FunctionCrossProductTest() {}
    FunctionCrossProductTest(Func1 f1, Func2 f2) : func1(f1), func2(f2) {
        int irtemp = std::max(this->func1.IRows(), this->func2.IRows());
        this->setIORows(irtemp, 3);

        this->set_input_domain(this->IRows(), { this->func1.input_domain(),
                                               this->func2.input_domain() });

        if (this->func1.ORows() != 3) {
            throw std::invalid_argument("Function 1 in cross product must have three output rows");
        }
        if (this->func2.ORows() != 3) {
            throw std::invalid_argument("Function 2 in cross product must have three output rows");
        }
        if (this->func1.IRows() != this->func2.IRows()) {
            throw std::invalid_argument("Functions 1,2 in cross product must have same numer of input rows");
        }

    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template<class Scalar, class T1, class T2>
    Vector3<Scalar> crossimpl(Scalar sign, ConstVectorBaseRef<T1> x1, ConstVectorBaseRef<T2> x2) const {
        Vector3<Scalar> out;
        out[0] = sign * (x1[1] * x2[2] - x1[2] * x2[1]);
        out[1] = sign * (x2[0] * x1[2] - x2[2] * x1[0]);
        out[2] = sign * (x1[0] * x2[1] - x1[1] * x2[0]);
        return out;
    }


    template <class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x,
        ConstVectorBaseRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        Eigen::MatrixBase<OutType>& fx = fx_.const_cast_derived();
        Vector3<Scalar> fx1;
        Vector3<Scalar> fx2;
       
        this->func1.compute(x, fx1);
        this->func2.compute(x, fx2);
        fx = crossimpl(Scalar(1.0), fx1, fx2);

    }

    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
        ConstVectorBaseRef<OutType> fx_,
        ConstMatrixBaseRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();
        MatrixBaseRef<JacType> jx = jx_.const_cast_derived();


        Vector3<Scalar> fx1;
        Vector3<Scalar> fx2;
        Eigen::Matrix<Scalar, 3, 3> cpm1;
        Eigen::Matrix<Scalar, 3, 3> cpm2;

        auto Impl = [&](auto& jx1, auto& jx2) {
            this->func1.compute_jacobian(x, fx1, jx1);
            this->func2.compute_jacobian(x, fx2, jx2);
            CrossProduct::cprodmat(fx2, cpm1, -1.0);
            CrossProduct::cprodmat(fx1, cpm2, 1.0);
            fx = crossimpl(Scalar(1.0), fx1, fx2);
            this->func1.right_jacobian_product(jx, cpm1, jx1,
                DirectAssignment(),
                std::bool_constant<false>());
            this->func2.right_jacobian_product(jx, cpm2, jx2,
                PlusEqualsAssignment(),
                std::bool_constant<false>());
        };

        const int irows = this->IRows();
        using JType = Eigen::Matrix<Scalar, 3, Base::IRC>;

        MemoryManager::allocate_run(irows, 
            Impl,
            TempSpec<JType>(3, irows),
            TempSpec<JType>(3, irows));
    }

    template <class InType, class OutType, class JacType, class AdjGradType,
        class AdjHessType, class AdjVarType>
        inline void compute_jacobian_adjointgradient_adjointhessian_impl(
            ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
            ConstMatrixBaseRef<JacType> jx_, ConstVectorBaseRef<AdjGradType> adjgrad_,
            ConstMatrixBaseRef<AdjHessType> adjhess_,
            ConstVectorBaseRef<AdjVarType> adjvars) const {
        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();
        MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
        VectorBaseRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
        MatrixBaseRef<AdjHessType> adjhess = adjhess_.const_cast_derived();

        Vector3<Scalar> fx1;
        Vector3<Scalar> fx2;

        Eigen::Matrix<Scalar, 3, 3> cpm1;
        Eigen::Matrix<Scalar, 3, 3> cpm2;
        Eigen::Matrix<Scalar, 3, 3> lcpm1;
        Eigen::Matrix<Scalar, 3, 3> lcpm2;


        auto Impl = [&](auto& jx1, auto& jx2, auto& jttemp, auto& gx2, auto& hx2) {
            this->func1.compute(x, fx1);
            this->func2.compute(x, fx2);

            CrossProduct::cprodmat(fx2, cpm1, -1.0);
            CrossProduct::cprodmat(fx1, cpm2, 1.0);

            CrossProduct::cprodmat(adjvars, lcpm1, 1.0);
            CrossProduct::cprodmat(adjvars, lcpm2, -1.0);

            Vector3<Scalar> adjt = adjvars;

            fx = crossimpl(Scalar(1.0), fx1, fx2);
            Vector3<Scalar> adjcross1 = crossimpl(Scalar(1.0), fx2, adjt);
            Vector3<Scalar> adjcross2 = crossimpl(Scalar(-1.0), fx1, adjt);

            fx1.setZero();
            fx2.setZero();

            this->func1.compute_jacobian_adjointgradient_adjointhessian(
                x, fx1, jx1, adjgrad, adjhess, adjcross1);
            this->func2.compute_jacobian_adjointgradient_adjointhessian(
                x, fx2, jx2, gx2, hx2, adjcross2);

            this->func2.accumulate_gradient(adjgrad, gx2, PlusEqualsAssignment());
            this->func2.accumulate_hessian(adjhess, hx2, PlusEqualsAssignment());

            this->func2.zero_matrix_domain(hx2);
            
            this->func2.right_jacobian_product(
                jttemp, lcpm2, jx2, DirectAssignment(),
                std::bool_constant<false>());
            this->func1.right_jacobian_product(
                hx2, jttemp.transpose(), jx1,
                DirectAssignment(), std::bool_constant<false>());

            // adjhess += hx2 + hx2^T 
            // func1 does this because hx2 now has its domain structure 
            this->func1.accumulate_product_hessian(adjhess, hx2);

            this->func1.right_jacobian_product(jx, cpm1, jx1,
                DirectAssignment(),
                std::bool_constant<false>());
            this->func2.right_jacobian_product(
                jx, cpm2, jx2, PlusEqualsAssignment(),
                std::bool_constant<false>());
        };


        const int irows = this->IRows();

        using JType = Eigen::Matrix<Scalar, 3, Base::IRC>;
        using JTTran = Eigen::Matrix<Scalar, Base::IRC, 3>;

        using GType = Func2_gradient<Scalar>;
        using HType = Func2_hessian<Scalar>;

        MemoryManager::allocate_run(irows, Impl,
            TempSpec<JType>(3, irows),
            TempSpec<JType>(3, irows),
            TempSpec<JType>(3, irows),
            TempSpec<GType>(irows, 1),
            TempSpec<HType>(irows, irows)
        );

    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
};





////////////////////////////////////////////////////////////////////////////////
template<class T>
using TS = TempSpec<T>;






int main() {
    using std::cin;
    using std::cout;
    using std::endl;



    using V1 = Vector<double, 1>;
    using V2 = Vector<double, 2>;
    using VX = Vector<double, -1>;
    using M2 = Eigen::Matrix<double, 2, 2>;
    using M4 = Eigen::Matrix<double, 4, 4>;
    using MX = Eigen::Matrix<double, -1, -1>;


    auto printlam = [&](auto&& mm) {
        std::cout << "Rows: " << mm.rows() << " Cols: " << mm.cols() << endl;
        std::cout << "RowsC: " << mm.RowsAtCompileTime << " ColsC: " << mm.ColsAtCompileTime << endl;
        std::cout << "RowsMC: " << mm.MaxRowsAtCompileTime << " ColsMC: " << mm.MaxColsAtCompileTime << endl;
        std::cout << mm << endl << endl;
        mm.setOnes();

    };

    auto printlam2 = [&](auto&& mm) {
        std::cout << "  Rows: " << mm.rows() << " Cols: " << mm.cols() << endl;
        std::cout << "  RowsC: " << mm.RowsAtCompileTime << " ColsC: " << mm.ColsAtCompileTime << endl;
        std::cout << "  RowsMC: " << mm.MaxRowsAtCompileTime << " ColsMC: " << mm.MaxColsAtCompileTime << endl;
        std::cout << mm << endl << endl;
        mm.setOnes();
    };

    auto printlam3 = [&](auto&& mm) {
        std::cout << "    Rows: " << mm.rows() << " Cols: " << mm.cols() << endl;
        std::cout << "    RowsC: " << mm.RowsAtCompileTime << " ColsC: " << mm.ColsAtCompileTime << endl;
        std::cout << "    RowsMC: " << mm.MaxRowsAtCompileTime << " ColsMC: " << mm.MaxColsAtCompileTime << endl;
        std::cout << mm << endl << endl;
        mm.setOnes();

    };

    auto PrintLam = [&](auto&& ... ms) {
        (printlam3(ms), ...);
    };
    auto PrintLam2 = [&](auto&& ... ms) {
        MemoryManager::allocate_run(10, PrintLam, TS<MX>(4, 4));
        (printlam2(ms), ...);
        MemoryManager::allocate_run(10, PrintLam, TS<MX>(34, 4));

    };
    auto PrintLam3 = [&](auto&& ... ms) {
        MemoryManager::allocate_run(10, PrintLam2, TS<MX>(4, 4));
        (printlam(ms), ...);
        MemoryManager::allocate_run(10, PrintLam2, TS<MX>(4, 4));

    };

    /*MemoryManager::allocate_run(10, PrintLam3, TS<MX>(7, 7));
    cout << "------------------------------------------------------------------------------" << endl;
    MemoryManager::allocate_run(10, PrintLam3, TS<MX>(11, 11));
    MemoryManager::allocate_run(10, PrintLam3, TS<MX>(6, 6));
    MemoryManager::allocate_run(10, PrintLam3, TS<MX>(30, 40));
    MemoryManager::allocate_run(10, PrintLam3, TS<MX>(30, 40));




    cin.get();*/




    using Gen = GenericFunction<-1, -1>;

    int n = 10;
    auto X = Arguments<-1>(n);
    auto v1 = X.head<3>().normalized();
    auto v2 = X.segment<3, 3>().normalized();
    auto v3 = X.segment<3, 6>().normalized();

    auto f11 = Gen(FunctionCrossProductTest{ v2, v1 });
    auto f12 = Gen(FunctionCrossProductTest{ f11,v2 });
    auto f13 = Gen(FunctionCrossProductTest{ f12,v3 });
    auto f14 = Gen(FunctionCrossProductTest{ f13,v3 });

    auto F1 = f14;

    auto f21 = Gen(v2.cross(v1));
    auto f22 = Gen(f21.cross(v2));
    auto f23 = Gen(f22.cross(v3));
    auto f24 = Gen(f23.cross(v3));

    auto F2 = f24;


    VectorX<double> x(n);
    x.setZero();
    x[0] = 1;
    x[1] = 1;
    x[4] = 2;
    x[5] = 1;
    x[7] = 1;

    Vector3<double> L;
    L.setOnes();
    //cout << F1.adjointhessian(x, L)<<endl<<endl;
    //cout << F1.adjointhessian(x, L) << endl << endl;
    //cout << F1.adjointhessian(x, L) << endl << endl;
    //cout << F2.adjointhessian(x, L) << endl << endl;

    //F2.SuperTest(x, 1000000);
    //std::vector<double> xx(12, 1.0);
    //cin.get();
    //MemoryManager::disable_arena_memory();

    auto Truth = F2.adjointhessian(x, L);
    //cout << Truth - F1.adjointhessian(x, L) << endl << endl;

    auto pool = ctpl::ThreadPool(8);

    std::mutex coutex;




    auto Job1 = [&](int id, int t) {
        double error = 0.0;
        VectorX<DefaultSuperScalar> xs = x.cast<DefaultSuperScalar>();
        VectorX<DefaultSuperScalar> ls = L.cast<DefaultSuperScalar>();
        Eigen::Matrix< DefaultSuperScalar, -1, -1> ht(n, n);
        Eigen::Matrix< DefaultSuperScalar, -1, -1> jt(3, n);
        Eigen::Matrix< DefaultSuperScalar, -1, 1> gt(n, 1);
        Eigen::Matrix< DefaultSuperScalar, 3, 1> fx;

        ht.setZero();
        for (int i = 0; i < 10000000 / 2; i++) {
            //F1.adjointhessian(xs, ht, ls);
           ht.setZero();
           jt.setZero();
           gt.setZero();
           jt.setZero();

           F1.compute_jacobian_adjointgradient_adjointhessian(xs,fx,jt,gt, ht, ls);
            //error += (M - Truth.cast<DefaultSuperScalar>()).squaredNorm()[0];
           
        }
        return error;
    };
    

    auto Job2 = [&](int id, int t) {
        double error = 0.0;
        VectorX<DefaultSuperScalar> xs = x.cast<DefaultSuperScalar>();
        VectorX<DefaultSuperScalar> ls = L.cast<DefaultSuperScalar>();
        Eigen::Matrix< DefaultSuperScalar, -1, -1> ht(n, n);
        Eigen::Matrix< DefaultSuperScalar, -1, -1> jt(3, n);
        Eigen::Matrix< DefaultSuperScalar, -1, 1> gt(n, 1);
        Eigen::Matrix< DefaultSuperScalar, 3, 1> fx;

        ht.setZero();
        for (int i = 0; i < 10000000 / 2; i++) {
            //F1.adjointhessian(xs, ht, ls);
            ht.setZero();
            jt.setZero();
            gt.setZero();
            jt.setZero();

            F2.compute_jacobian_adjointgradient_adjointhessian(xs, fx, jt, gt, ht, ls);
            //error += (M - Truth.cast<DefaultSuperScalar>()).squaredNorm()[0];

        }
        return error;
    };


    int nt = 8;
    Eigen::BenchTimer t1;
    std::vector<std::future<double>> results(nt);
    t1.start();
    //for (int i = 0; i < nt; i++) { results[i] = pool.push(Job1, i); }
    //for (int i = 0; i < nt; i++) cout << results[i].get() << endl;
    Job1(1, 1);
    t1.stop();
    std::cout << endl << "Time: " << t1.value() * 1000.0 << endl;

    

    //MemoryManager::disable_arena_memory();

   /* t1.start();
    for (int i = 0; i < nt; i++) { results[i] = pool.push(Job1, i); }
    for (int i = 0; i < nt; i++) cout << results[i].get() << endl;
    t1.stop();
    std::cout << endl << "Time: " << t1.value() * 1000.0 << endl;


    t1.start();
    for (int i = 0; i < nt; i++) { results[i] = pool.push(Job1, i); }
    for (int i = 0; i < nt; i++) cout << results[i].get() << endl;
    t1.stop();
    std::cout << endl << "Time: " << t1.value() * 1000.0 << endl;*/


   // cin.get();

    return 0;
}
