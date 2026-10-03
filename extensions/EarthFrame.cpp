#include "ASSET_Extensions.h"
#include "sofa.h"

#include <cmath>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace ASSET {

/** Reference-compatible hourly-cached ICRF-to-ECEF rotation.
 *
 * The implementation uses intact IAU SOFA 2023-10-11 routines for IAU
 * 2006/2000A bias-precession-nutation, CIO locator, and polar motion. EOP data
 * are sampled by UTC day, matching the released Fortran implementation. Slow
 * terms and the equation of origins are held at the start of each UTC hour;
 * Earth rotation is evaluated at every call. Input is elapsed seconds from the
 * constructor's UTC Julian-date epoch. Output is a row-major 3x3 C_ECEF<-ICRF.
 *
 * Time derivatives are exact within an hourly cache interval because only ERA
 * varies there. The reference model jumps at hour boundaries; derivatives are
 * therefore not defined at those boundaries. This is a SOFA-based derived work
 * and does not itself constitute software provided or endorsed by SOFA.
 */
struct EarthFrame : VectorFunction<EarthFrame, 1, 9, AutodiffFwd, AutodiffFwd> {
  using Base = VectorFunction<EarthFrame, 1, 9, AutodiffFwd, AutodiffFwd>;
  DENSE_FUNCTION_BASE_TYPES(Base)
  static constexpr bool IsVectorizable = false;

  struct Eop { double xp_arcsec, yp_arcsec, dut1_seconds; };
  struct HourData {
    double rbpn[3][3], rpom[3][3], equation_of_origins, dut1_seconds;
  };
  double epoch_jd_utc;
  std::string eop_file;
  std::map<int, Eop> eop;
  mutable std::map<long long, HourData> cache;
  mutable std::mutex cache_mutex;

  EarthFrame(double epoch, const std::string& path)
      : epoch_jd_utc(epoch), eop_file(path) {
    if (!std::isfinite(epoch))
      throw std::invalid_argument("EarthFrame epoch_jd_utc must be finite");
    std::ifstream input(path);
    if (!input) throw std::invalid_argument("Cannot open EarthFrame EOP file: " + path);
    std::string line;
    while (std::getline(input, line)) {
      std::istringstream row(line);
      int year, month, day, mjd;
      Eop record;
      if (row >> year >> month >> day >> mjd >> record.xp_arcsec
              >> record.yp_arcsec >> record.dut1_seconds)
        eop[mjd] = record;
    }
    if (eop.empty()) throw std::invalid_argument("EarthFrame EOP file contains no records");
  }
  EarthFrame(const EarthFrame& other)
      : epoch_jd_utc(other.epoch_jd_utc), eop_file(other.eop_file), eop(other.eop) {}

  HourData make_hour(double hour_jd) const {
    constexpr double day_seconds = 86400.0;
    constexpr double arcsec_to_rad = 4.848136811095359935899141e-6;
    HourData data{};
    const int mjd = int(std::floor(hour_jd - 2400000.5));
    const auto found = eop.find(mjd);
    if (found == eop.end())
      throw std::out_of_range("EarthFrame EOP data do not cover requested UTC day (MJD "
                              + std::to_string(mjd) + ")");
    const double xp = found->second.xp_arcsec * arcsec_to_rad;
    const double yp = found->second.yp_arcsec * arcsec_to_rad;
    data.dut1_seconds = found->second.dut1_seconds;
    int iy, im, id; double fraction;
    if (iauJd2cal(2400000.5, hour_jd - 2400000.5, &iy, &im, &id, &fraction))
      throw std::invalid_argument("EarthFrame epoch is outside the SOFA calendar range");
    double delta_at;
    if (iauDat(iy, im, id, fraction, &delta_at) < 0)
      throw std::invalid_argument("EarthFrame epoch is invalid for leap-second conversion");
    const double tt = hour_jd + (delta_at + 32.184) / day_seconds;
    iauPnm06a(2400000.5, tt - 2400000.5, data.rbpn);
    double x, y;
    iauBpn2xy(data.rbpn, &x, &y);
    const double s = iauS06(2400000.5, tt - 2400000.5, x, y);
    data.equation_of_origins = iauEors(data.rbpn, s);
    const double sp = iauSp00(2400000.5, tt - 2400000.5);
    // Preserve the released model's explicit P = R3(-s') R2(xp) R1(yp)
    // convention. It is the inverse-sign convention of SOFA iauPom00, so use
    // SOFA's rotation primitives rather than silently changing reference
    // trajectory behavior.
    iauIr(data.rpom);
    iauRx(yp, data.rpom);
    iauRy(xp, data.rpom);
    iauRz(-sp, data.rpom);
    return data;
  }

  const HourData& hour_data(long long hour_index, double hour_jd) const {
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto found = cache.find(hour_index);
    if (found == cache.end()) found = cache.emplace(hour_index, make_hour(hour_jd)).first;
    return found->second;
  }

  template<class InType, class OutType>
  inline void compute_impl(ConstVectorBaseRef<InType> input,
                           ConstVectorBaseRef<OutType> output_) const {
    using Scalar = typename InType::Scalar;
    auto& output = output_.const_cast_derived();
    const double elapsed_value = static_cast<double>(autodiff::val(input[0]));
    if (!std::isfinite(elapsed_value))
      throw std::invalid_argument("EarthFrame elapsed seconds must be finite");
    const double jd_value = epoch_jd_utc + elapsed_value/86400.0;
    const double day_jd = std::floor(jd_value - 0.5) + 0.5;
    const int hour_of_day = int(std::floor((jd_value-day_jd)*24.0));
    const double hour_jd = day_jd + double(hour_of_day)/24.0;
    const long long hour_index = std::llround((hour_jd-2400000.5)*24.0);
    const HourData& slow = hour_data(hour_index, hour_jd);

    // Unwrapped ERA is sufficient because only sin/cos are used. Avoiding a
    // modulo operation preserves automatic derivatives within the hour.
    const Scalar jd_utc = Scalar(epoch_jd_utc) + input[0]/Scalar(86400.0);
    const Scalar tu = jd_utc + Scalar(slow.dut1_seconds/86400.0) - Scalar(2451545.0);
    const Scalar era = Scalar(2.0*M_PI) *
        (Scalar(0.7790572732640) + Scalar(1.00273781191135448)*tu);
    const Scalar angle = era - Scalar(slow.equation_of_origins);
    const Scalar c = cos(angle), s = sin(angle);
    Eigen::Matrix<Scalar,3,3> rz, rbpn, rpom;
    rz << c,s,0, -s,c,0, 0,0,1;
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
      rbpn(i,j)=Scalar(slow.rbpn[i][j]); rpom(i,j)=Scalar(slow.rpom[i][j]);
    }
    const Eigen::Matrix<Scalar,3,3> result = rpom * rz * rbpn;
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) output[3*i+j]=result(i,j);
  }

  static void Build(py::module& m, const char* name) {
    auto obj=py::class_<EarthFrame>(m,name,
      "Hourly-cached IAU 2006/2000A ICRF-to-ECEF matrix from elapsed UTC seconds.");
    obj.def(py::init<double,const std::string&>(),py::arg("epoch_jd_utc"),py::arg("eop_file"));
    obj.def_property_readonly("epoch_jd_utc",[](const EarthFrame& f){return f.epoch_jd_utc;});
    obj.def_property_readonly("eop_file",[](const EarthFrame& f){return f.eop_file;});
    Base::DenseBaseBuild(obj);
    obj.def("__call__",[](const EarthFrame& f,const GenericFunction<-1,-1>& input){
      return GenericFunction<-1,-1>(f.eval(input));
    },py::arg("elapsed_seconds"));
  }
};

void BuildEarthFrame(FunctionRegistry& reg, py::module& m) {
  reg.Build_Register<EarthFrame>(m,"EarthFrame");
}
} // namespace ASSET
