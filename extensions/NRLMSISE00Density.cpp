#include "ASSET_Extensions.h"
#include "NumericalDensityDerivatives.h"
extern "C" {
#include "nrlmsise-00.h"
}
#include "sofa.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace ASSET {

/** Direct NRLMSISE-00 effective drag density.
 *
 * Inputs are [WGS84 height_km, latitude_rad, longitude_rad,
 * elapsed_UTC_seconds]. Output is density [kg/m^3]. The UTC epoch, default
 * solar/geomagnetic indices, and optional CelesTrak space-weather file are
 * constructor data. The default cutoff is 1000 km, matching the released
 * propagator; callers may raise it when an outer smooth taper is composed.
 * ASSET's vendored C adapter makes the legacy mutable
 * scratch arrays thread-local, so independent evaluations are reentrant.
 *
 * The legacy empirical model is double-only. Centered physical-coordinate
 * differences provide its local gradient and Hessian for ASSET phases. Date
 * and weather bins remain nonsmooth and must not be placed on collocation
 * nodes. A configurable cutoff permits smooth outer tapering above 1000 km.
 */
struct NRLMSISE00Density : VectorFunction<NRLMSISE00Density, 4, 1> {
  using Base = VectorFunction<NRLMSISE00Density, 4, 1>;
  DENSE_FUNCTION_BASE_TYPES(Base)
  static constexpr bool IsVectorizable = false;
  struct Weather { double f107, f107a, ap; };

  double epoch_jd_utc, default_f107, default_f107a, default_ap, cutoff_km;
  Eigen::Vector4d derivative_steps{0.05, 1.0e-4, 1.0e-4, 10.0};
  std::string weather_file;
  std::map<int,Weather> weather;

  NRLMSISE00Density(double epoch, double f107, double f107a, double ap,
                    const std::string& weather_path, double cutoff)
      : epoch_jd_utc(epoch), default_f107(f107), default_f107a(f107a),
        default_ap(ap), cutoff_km(cutoff), weather_file(weather_path) {
    if (!std::isfinite(epoch) || !std::isfinite(f107) || !std::isfinite(f107a) ||
        !std::isfinite(ap) || !std::isfinite(cutoff) || f107 < 0 ||
        f107a < 0 || ap < 0 || cutoff <= 0)
      throw std::invalid_argument("NRLMSISE00Density epoch and indices must be finite and nonnegative");
    if (!weather_path.empty()) load_weather(weather_path);
  }

  void load_weather(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::invalid_argument("Cannot open space-weather file: " + path);
    std::string line;
    while (std::getline(input,line)) {
      if (line.empty() || line[0]<'0' || line[0]>'9') continue;
      std::istringstream row(line); std::vector<double> values; double value;
      while (row >> value) values.push_back(value);
      if (values.size() < 32) continue;
      const int year=int(values[0]), month=int(values[1]), day=int(values[2]);
      const int doy=day_of_year(year,month,day);
      // The CelesTrak format ends with observed F10.7, centered F10.7A, and
      // last F10.7. This is also how the released Fortran selects its fields.
      weather[year*1000+doy] = {values[values.size()-3],
                                values[values.size()-2], values[22]};
    }
    if (weather.empty()) throw std::invalid_argument("Space-weather file contains no usable daily records");
  }

  static bool leap(int y) { return (y%4==0 && y%100!=0) || y%400==0; }
  static int day_of_year(int y,int m,int d) {
    constexpr int days[12]={31,28,31,30,31,30,31,31,30,31,30,31};
    int result=d; for(int i=1;i<m;++i) result += days[i-1] + (i==2 && leap(y));
    return result;
  }

  double density(const Eigen::Vector4d& x) const {
    if (!x.allFinite()) throw std::invalid_argument("NRLMSISE00Density inputs must be finite");
    if (x[0] >= cutoff_km) return 0.0;
    const double jd=epoch_jd_utc+x[3]/86400.0;
    int year,month,day; double fraction;
    if (iauJd2cal(2400000.5,jd-2400000.5,&year,&month,&day,&fraction))
      throw std::invalid_argument("NRLMSISE00Density epoch is outside the calendar range");
    const int doy=day_of_year(year,month,day), weather_day=year*1000+doy;
    double f107=default_f107,f107a=default_f107a,ap=default_ap;
    const auto found=weather.find(weather_day);
    if(found!=weather.end()) { f107=found->second.f107; f107a=found->second.f107a; ap=found->second.ap; }
    nrlmsise_input in{}; nrlmsise_flags flags{}; nrlmsise_output out{}; ap_array ap_values{};
    flags.switches[0]=1; for(int i=1;i<24;++i) flags.switches[i]=1;
    for(double& item:ap_values.a) item=ap;
    in.year=year; in.doy=doy; in.sec=fraction*86400.0; in.alt=x[0];
    in.g_lat=x[1]*180.0/M_PI; in.g_long=x[2]*180.0/M_PI;
    in.lst=std::fmod(in.sec/3600.0+x[2]*12.0/M_PI,24.0); if(in.lst<0) in.lst+=24.0;
    in.f107A=f107a; in.f107=f107; in.ap=ap; in.ap_a=&ap_values;
    gtd7d(&in,&flags,&out);
    return std::max(0.0,out.d[5]);
  }

  Eigen::VectorXd density_batch(const Eigen::Ref<const Eigen::MatrixXd>& points,
                                int threads) const {
    if (points.cols() != 4)
      throw std::invalid_argument("NRLMSISE00Density.density_batch expects N by 4 input");
    if (threads < 0)
      throw std::invalid_argument("NRLMSISE00Density.density_batch threads must be nonnegative");
    Eigen::VectorXd result(points.rows());
    const int count = static_cast<int>(points.rows());
    if (threads == 1 || count < 2) {
      for (int i=0;i<count;++i) result[i]=density(points.row(i).transpose());
      return result;
    }
#ifdef _OPENMP
    const int requested = threads == 0 ? omp_get_max_threads() : threads;
#pragma omp parallel for schedule(static) num_threads(requested)
#endif
    for (int i=0;i<count;++i) result[i]=density(points.row(i).transpose());
    return result;
  }

  template<class InType,class OutType>
  void compute_impl(ConstVectorBaseRef<InType> x,ConstVectorBaseRef<OutType> fx_) const {
    Eigen::Vector4d values; for(int i=0;i<4;++i) values[i]=x[i];
    fx_.const_cast_derived()[0]=density(values);
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
    auto obj=py::class_<NRLMSISE00Density>(m,name,
      "Direct NRLMSISE-00 density: [height_km,lat_rad,lon_rad,elapsed_s] -> kg/m^3.");
    obj.def(py::init<double,double,double,double,const std::string&,double>(),py::arg("epoch_jd_utc"),
      py::arg("f107")=150.0,py::arg("f107a")=150.0,py::arg("ap")=4.0,py::arg("weather_file")="",
      py::arg("cutoff_km")=1000.0);
    obj.def_property_readonly("supports_derivatives",[](const NRLMSISE00Density&){return true;});
    obj.def_readonly("cutoff_km",&NRLMSISE00Density::cutoff_km);
    obj.def_property_readonly("weather_file",[](const NRLMSISE00Density& f){return f.weather_file;});
    obj.def("density_batch", &NRLMSISE00Density::density_batch,
            py::arg("points"), py::arg("threads")=0,
            py::call_guard<py::gil_scoped_release>(),
            "Evaluate an N by 4 array; threads=0 uses the OpenMP maximum.");
    Base::DenseBaseBuild(obj);
    obj.def("__call__",[](const NRLMSISE00Density& f,const GenericFunction<-1,-1>& input){
      return GenericFunction<-1,-1>(f.eval(input));
    },py::arg("geodetic_and_time"));
  }
};

void BuildNRLMSISE00Density(FunctionRegistry& reg,py::module& m){
  reg.Build_Register<NRLMSISE00Density>(m,"NRLMSISE00Density");
}
} // namespace ASSET
