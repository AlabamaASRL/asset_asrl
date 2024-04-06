#include <bind/VectorFunctions/CommonFunctions/BindInterpTable4D.h>

void ASSET::BindInterpTable4D(py::module& m) {

  auto obj = py::class_<InterpTable4D, std::shared_ptr<InterpTable4D>>(m, "InterpTable4D");

  obj.def(py::init<const Eigen::VectorXd&,
                   const Eigen::VectorXd&,
                   const Eigen::VectorXd&,
                   const Eigen::VectorXd&,
                   const Eigen::Tensor<double, 4>&,
                   std::string,
                   bool>(),
          py::arg("xs"),
          py::arg("ys"),
          py::arg("zs"),
          py::arg("ws"),
          py::arg("fs"),
          py::arg("kind") = std::string("cubic"),
          py::arg("cache") = false);

  obj.def("interp", py::overload_cast<double, double, double, double>(&InterpTable4D::interp, py::const_));
  obj.def("interp_deriv1",
          py::overload_cast<double, double, double, double>(&InterpTable4D::interp_deriv1, py::const_));
  obj.def("interp_deriv2",
          py::overload_cast<double, double, double, double>(&InterpTable4D::interp_deriv2, py::const_));

  obj.def_readwrite("WarnOutOfBounds", &InterpTable4D::WarnOutOfBounds);
  obj.def_readwrite("ThrowOutOfBounds", &InterpTable4D::ThrowOutOfBounds);

  obj.def("__call__",
          py::overload_cast<double, double, double, double>(&InterpTable4D::interp, py::const_),
          py::is_operator());

  obj.def("__call__",
          [](std::shared_ptr<InterpTable4D>& self,
             const GenericFunction<-1, 1>& x,
             const GenericFunction<-1, 1>& y,
             const GenericFunction<-1, 1>& z,
             const GenericFunction<-1, 1>& w) {
            return GenericFunction<-1, 1>(InterpFunction4D(self).eval(stack(x, y, z, w)));
          });

  obj.def("__call__",
          [](std::shared_ptr<InterpTable4D>& self,
             const Segment<-1, 1, -1>& x,
             const Segment<-1, 1, -1>& y,
             const Segment<-1, 1, -1>& z,
             const Segment<-1, 1, -1>& w

          ) { return GenericFunction<-1, 1>(InterpFunction4D(self).eval(stack(x, y, z, w))); });

  obj.def("__call__", [](std::shared_ptr<InterpTable4D>& self, const Segment<-1, -1, -1>& xyzw) {
    return GenericFunction<-1, 1>(InterpFunction4D(self).eval(xyzw));
  });

  obj.def("__call__", [](std::shared_ptr<InterpTable4D>& self, const GenericFunction<-1, -1>& xyzw) {
    return GenericFunction<-1, 1>(InterpFunction4D(self).eval(xyzw));
  });

  obj.def("sf", [](std::shared_ptr<InterpTable4D>& self) {
    return GenericFunction<-1, 1>(InterpFunction4D(self));
  });
  obj.def("vf", [](std::shared_ptr<InterpTable4D>& self) {
    return GenericFunction<-1, -1>(InterpFunction4D(self));
  });

  m.def("InterpTable4DSpeedTest",
        [](const GenericFunction<-1, 1>& tabf,
           double xl,
           double xu,
           double yl,
           double yu,
           double zl,
           double zu,
           double wl,
           double wu,

           int nsamps,
           bool lin) {
          Eigen::ArrayXd xsamps;
          xsamps.setRandom(nsamps);
          xsamps += 1;
          xsamps /= 2;
          xsamps *= (xu - xl);
          xsamps += xl;

          Eigen::ArrayXd ysamps;
          ysamps.setRandom(nsamps);
          ysamps += 1;
          ysamps /= 2;
          ysamps *= (yu - yl);
          ysamps += yl;

          Eigen::ArrayXd zsamps;
          zsamps.setRandom(nsamps);
          zsamps += 1;
          zsamps /= 2;
          zsamps *= (zu - zl);
          zsamps += zl;

          Eigen::ArrayXd wsamps;
          wsamps.setRandom(nsamps);
          wsamps += 1;
          wsamps /= 2;
          wsamps *= (wu - wl);
          wsamps += wl;

          if (lin) {
            xsamps.setLinSpaced(xl, xu);
            ysamps.setLinSpaced(yl, yu);
            zsamps.setLinSpaced(zl, zu);
            wsamps.setLinSpaced(wl, wu);
          }

          Eigen::VectorXd xyzw(4);
          Vector1<double> f;
          f.setZero();

          Utils::Timer Runtimer;
          Runtimer.start();

          double tmp = 0;
          for (int i = 0; i < nsamps; i++) {

            xyzw[0] = xsamps[i];
            xyzw[1] = ysamps[i];
            xyzw[2] = zsamps[i];
            xyzw[3] = wsamps[i];

            tabf.compute(xyzw, f);
            tmp += f[0] / double(i + 3);

            f.setZero();
          }
          Runtimer.stop();
          double tseconds = double(Runtimer.count<std::chrono::microseconds>()) / 1000000;
          fmt::print("Total Time: {0:} ms \n", tseconds * 1000);

          return tmp;
        });
}
