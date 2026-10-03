#include "ASSET_Extensions.h"
#include "NumericalDensityDerivatives.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" void asset_jb2008(
    double amjd, double sun_ra, double sun_dec, double sat_ra,
    double sat_lat, double sat_alt, double f10, double f10b, double s10,
    double s10b, double xm10, double xm10b, double y10, double y10b,
    double dstdtc, double* tinf, double* tlocal, double* rho);

namespace ASSET {

/** Reentrant adapter for the public JB2008 integration-form model.
 *
 * Input is [height_km, geodetic_latitude_rad, longitude_rad,
 * elapsed_seconds]; output is kg/m^3.  Solar geometry and sidereal angle
 * reproduce the released Fortran propagator.  The JB2008 reference contains
 * only local working arrays and read-only DATA constants; it is compiled with
 * recursive local storage so calls at independent phase/mesh nodes may run in
 * parallel.  Solar proxies are constructor values until a dedicated JB2008
 * SOLFSMY/DTC data provider is added. Because the reference is double-only,
 * centered differences in physical coordinates provide the ASSET gradient
 * and symmetric Hessian. Nodes must stay away from its cutoff and from time
 * bins in any future operational index provider.
 */
struct JB2008Density : VectorFunction<JB2008Density,4,1> {
  using Base = VectorFunction<JB2008Density,4,1>;
  DENSE_FUNCTION_BASE_TYPES(Base)
  static constexpr bool IsVectorizable = false;

  double epoch_jd_utc, f10, f10b, s10, s10b, xm10, xm10b, y10, y10b;
  double dstdtc, cutoff_km;
  Eigen::Vector4d derivative_steps{0.05, 1.0e-4, 1.0e-4, 10.0};

  JB2008Density(double epoch, double f10_, double f10b_, double s10_,
                double s10b_, double xm10_, double xm10b_, double y10_,
                double y10b_, double dstdtc_, double cutoff)
      : epoch_jd_utc(epoch), f10(f10_), f10b(f10b_), s10(s10_),
        s10b(s10b_), xm10(xm10_), xm10b(xm10b_), y10(y10_),
        y10b(y10b_), dstdtc(dstdtc_), cutoff_km(cutoff) {
    const double values[] = {epoch,f10_,f10b_,s10_,s10b_,xm10_,xm10b_,
                             y10_,y10b_,dstdtc_,cutoff};
    for(double value:values) if(!std::isfinite(value))
      throw std::invalid_argument("JB2008Density parameters must be finite");
    if(f10_<0 || f10b_<0 || s10_<0 || s10b_<0 || xm10_<0 || xm10b_<0 ||
       y10_<0 || y10b_<0 || cutoff<=0)
      throw std::invalid_argument("JB2008Density indices must be nonnegative and cutoff positive");
  }

  static void sun_position(double amjd, double& right_ascension,
                           double& declination) {
    constexpr double pi=3.141592653589793238462643383279502884;
    const double d2000=amjd-51544.5;
    const double mean_anomaly=(357.528+0.9856003*d2000)*pi/180.0;
    double mean_longitude=std::fmod(280.460+0.9856474*d2000,360.0);
    if(mean_longitude<0) mean_longitude+=360.0;
    const double ecliptic=(mean_longitude+1.915*std::sin(mean_anomaly)+
                           0.020*std::sin(2.0*mean_anomaly))*pi/180.0;
    const double obliquity=(23.439-0.0000004*d2000)*pi/180.0;
    const double t2=std::pow(std::tan(0.5*obliquity),2);
    right_ascension=ecliptic-t2*std::sin(2*ecliptic)+
                    0.5*t2*t2*std::sin(4*ecliptic);
    right_ascension=std::fmod(right_ascension,2*pi);
    if(right_ascension<0) right_ascension+=2*pi;
    declination=std::asin(std::sin(obliquity)*std::sin(ecliptic));
  }

  static double gmst(double jd) {
    constexpr double two_pi=6.283185307179586476925286766559;
    double turns=std::fmod(0.7790572732640+
                           1.00273781191135448*(jd-2451545.0),1.0);
    if(turns<0) turns+=1.0;
    return two_pi*turns;
  }

  double density(const Eigen::Vector4d& input) const {
    if(!input.allFinite())
      throw std::invalid_argument("JB2008Density inputs must be finite");
    if(input[0]>=cutoff_km) return 0.0;
    const double jd=epoch_jd_utc+input[3]/86400.0;
    const double amjd=jd-2400000.5;
    double sun_ra,sun_dec; sun_position(amjd,sun_ra,sun_dec);
    constexpr double two_pi=6.283185307179586476925286766559;
    double sat_ra=std::fmod(gmst(jd)+input[2],two_pi);
    if(sat_ra<0) sat_ra+=two_pi;
    double tinf,tlocal,rho;
    // JB2008's thermospheric integration is not defined below 90 km. Phase
    // line searches can temporarily cross path bounds, so hold the provider
    // at its lower domain edge rather than return NaN and abort the solver.
    const double model_altitude=std::max(90.0,input[0]);
    asset_jb2008(amjd,sun_ra,sun_dec,sat_ra,input[1],model_altitude,
                 f10,f10b,s10,s10b,xm10,xm10b,y10,y10b,dstdtc,
                 &tinf,&tlocal,&rho);
    if(!std::isfinite(rho))
      throw std::runtime_error("JB2008 returned a non-finite density");
    return std::max(0.0,rho);
  }

  Eigen::VectorXd density_batch(const Eigen::Ref<const Eigen::MatrixXd>& points,
                                int threads) const {
    if(points.cols()!=4)
      throw std::invalid_argument("JB2008Density.density_batch expects N by 4 input");
    if(threads<0)
      throw std::invalid_argument("JB2008Density.density_batch threads must be nonnegative");
    Eigen::VectorXd result(points.rows());
    const int count=static_cast<int>(points.rows());
    if(threads==1 || count<2) {
      for(int i=0;i<count;++i) result[i]=density(points.row(i).transpose());
      return result;
    }
#ifdef _OPENMP
    const int requested=threads==0 ? omp_get_max_threads() : threads;
#pragma omp parallel for schedule(static) num_threads(requested)
#endif
    for(int i=0;i<count;++i) result[i]=density(points.row(i).transpose());
    return result;
  }

  template<class InType,class OutType>
  void compute_impl(ConstVectorBaseRef<InType> input,
                    ConstVectorBaseRef<OutType> output_) const {
    Eigen::Vector4d values; for(int i=0;i<4;++i) values[i]=input[i];
    output_.const_cast_derived()[0]=density(values);
  }
  DensityDerivatives derivatives(const Eigen::Vector4d& x) const {
    return centered_density_derivatives(
        [this](const Eigen::Vector4d& point) { return density(point); },
        x, derivative_steps);
  }
  template<class InType,class OutType,class JacType>
  void compute_jacobian_impl(ConstVectorBaseRef<InType> x,ConstVectorBaseRef<OutType> fx_,ConstMatrixBaseRef<JacType> jx_) const {
    Eigen::Vector4d values; for(int i=0;i<4;++i) values[i]=x[i];
    const auto result=derivatives(values);
    fx_.const_cast_derived()[0]=result.value;
    jx_.const_cast_derived().row(0)=result.gradient.transpose();
  }
  template<class InType,class OutType,class JacType,class GradType,class HessType,class AdjType>
  void compute_jacobian_adjointgradient_adjointhessian_impl(ConstVectorBaseRef<InType> x,ConstVectorBaseRef<OutType> fx_,
      ConstMatrixBaseRef<JacType> jx_,ConstVectorBaseRef<GradType> grad_,ConstMatrixBaseRef<HessType> hess_,ConstVectorBaseRef<AdjType> adj) const {
    Eigen::Vector4d values; for(int i=0;i<4;++i) values[i]=x[i];
    const auto result=derivatives(values);
    fx_.const_cast_derived()[0]=result.value;
    jx_.const_cast_derived().row(0)=result.gradient.transpose();
    grad_.const_cast_derived()=result.gradient*adj[0];
    hess_.const_cast_derived()=result.hessian*adj[0];
  }

  static void Build(py::module& m,const char* name) {
    auto obj=py::class_<JB2008Density>(m,name,
      "Direct reentrant JB2008 density: [height,lat,lon,elapsed] -> kg/m^3.");
    obj.def(py::init<double,double,double,double,double,double,double,double,double,double,double>(),
      py::arg("epoch_jd_utc"),py::arg("f10")=150.0,py::arg("f10b")=150.0,
      py::arg("s10")=150.0,py::arg("s10b")=150.0,py::arg("xm10")=150.0,
      py::arg("xm10b")=150.0,py::arg("y10")=150.0,py::arg("y10b")=150.0,
      py::arg("dstdtc")=0.0,py::arg("cutoff_km")=1000.0);
    obj.def_property_readonly("supports_derivatives",[](const JB2008Density&){return true;});
    obj.def("density_batch",&JB2008Density::density_batch,py::arg("points"),
            py::arg("threads")=0,py::call_guard<py::gil_scoped_release>());
    Base::DenseBaseBuild(obj);
    obj.def("__call__",[](const JB2008Density& f,const GenericFunction<-1,-1>& input){
      return GenericFunction<-1,-1>(f.eval(input));
    },py::arg("geodetic_and_time"));
  }
};

void BuildJB2008Density(FunctionRegistry& reg,py::module& m) {
  reg.Build_Register<JB2008Density>(m,"JB2008Density");
}
}  // namespace ASSET
