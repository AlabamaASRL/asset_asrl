#include "ASSET_Extensions.h"

#include <cmath>
#include <stdexcept>

namespace ASSET {

/** Smoothly blend two density providers and taper the upper cutoff.
 *
 * Input is [height_km, rho_low_kg_m3, rho_high_kg_m3].  The low-altitude
 * provider is NRLMSISE-00 and the high-altitude provider is JB2008 in the
 * standard composition.  A tanh blend centered at blend_center_km avoids a
 * force discontinuity.  A second tanh can smoothly approach zero at the
 * configured upper cutoff; set cutoff_width_km=0 for the reference hard
 * cutoff.  Native automatic differentiation covers the blend itself.
 */
struct SmoothAtmosphereBlend
    : VectorFunction<SmoothAtmosphereBlend,3,1,AutodiffFwd,AutodiffFwd> {
  using Base=VectorFunction<SmoothAtmosphereBlend,3,1,AutodiffFwd,AutodiffFwd>;
  DENSE_FUNCTION_BASE_TYPES(Base)
  static constexpr bool IsVectorizable=false;

  double blend_center_km,blend_width_km,cutoff_km,cutoff_width_km;
  SmoothAtmosphereBlend(double center,double width,double cutoff,double cutoff_width)
      :blend_center_km(center),blend_width_km(width),cutoff_km(cutoff),
       cutoff_width_km(cutoff_width) {
    if(!std::isfinite(center)||!std::isfinite(width)||width<=0||
       !std::isfinite(cutoff)||cutoff<=center||
       !std::isfinite(cutoff_width)||cutoff_width<0)
      throw std::invalid_argument("SmoothAtmosphereBlend requires positive width, cutoff above center, and nonnegative cutoff width");
  }

  template<class InType,class OutType>
  void compute_impl(ConstVectorBaseRef<InType> input,
                    ConstVectorBaseRef<OutType> output_) const {
    using Scalar=typename InType::Scalar;
    const Scalar height=input[0],rho_low=input[1],rho_high=input[2];
    if(rho_low<Scalar(0.0)||rho_high<Scalar(0.0))
      throw std::invalid_argument("SmoothAtmosphereBlend densities must be nonnegative");
    const Scalar weight=Scalar(0.5)*(Scalar(1.0)+
      tanh((height-Scalar(blend_center_km))/Scalar(blend_width_km)));
    Scalar density=(Scalar(1.0)-weight)*rho_low+weight*rho_high;
    if(cutoff_width_km>0.0) {
      const Scalar taper=Scalar(0.5)*(Scalar(1.0)-
        tanh((height-Scalar(cutoff_km))/Scalar(cutoff_width_km)));
      density*=taper;
    } else if(height>=Scalar(cutoff_km)) {
      density=Scalar(0.0);
    }
    output_.const_cast_derived()[0]=density;
  }

  static void Build(py::module& m,const char* name) {
    auto obj=py::class_<SmoothAtmosphereBlend>(m,name,
      "Smooth NRLMSISE/JB2008 density blend: [height,rho_low,rho_high] -> rho.");
    obj.def(py::init<double,double,double,double>(),
      py::arg("blend_center_km")=600.0,py::arg("blend_width_km")=25.0,
      py::arg("cutoff_km")=1000.0,py::arg("cutoff_width_km")=10.0);
    obj.def_readonly("blend_center_km",&SmoothAtmosphereBlend::blend_center_km);
    obj.def_readonly("blend_width_km",&SmoothAtmosphereBlend::blend_width_km);
    obj.def_readonly("cutoff_km",&SmoothAtmosphereBlend::cutoff_km);
    obj.def_readonly("cutoff_width_km",&SmoothAtmosphereBlend::cutoff_width_km);
    Base::DenseBaseBuild(obj);
    obj.def("__call__",[](const SmoothAtmosphereBlend& f,const GenericFunction<-1,-1>& input){
      return GenericFunction<-1,-1>(f.eval(input));
    },py::arg("height_and_densities"));
  }
};

void BuildSmoothAtmosphereBlend(FunctionRegistry& reg,py::module& m) {
  reg.Build_Register<SmoothAtmosphereBlend>(m,"SmoothAtmosphereBlend");
}
}  // namespace ASSET
