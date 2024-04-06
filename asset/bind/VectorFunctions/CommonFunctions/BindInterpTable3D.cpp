#include <bind/VectorFunctions/CommonFunctions/BindInterpTable3D.h>

void ASSET::BindInterpTable3D(py::module& m) {

  auto obj = py::class_<InterpTable3D, std::shared_ptr<InterpTable3D>>(m, "InterpTable3D");

  obj.def(py::init<const Eigen::VectorXd&,
                   const Eigen::VectorXd&,
                   const Eigen::VectorXd&,
                   const Eigen::Tensor<double, 3>&,
                   std::string,
                   bool>(),
          py::arg("xs"),
          py::arg("ys"),
          py::arg("zs"),
          py::arg("fs"),
          py::arg("kind") = std::string("cubic"),
          py::arg("cache") = false);

  obj.def("interp", py::overload_cast<double, double, double>(&InterpTable3D::interp, py::const_));
  obj.def("interp_deriv1",
          py::overload_cast<double, double, double>(&InterpTable3D::interp_deriv1, py::const_));
  obj.def("interp_deriv2",
          py::overload_cast<double, double, double>(&InterpTable3D::interp_deriv2, py::const_));

  obj.def_readwrite("WarnOutOfBounds", &InterpTable3D::WarnOutOfBounds);
  obj.def_readwrite("ThrowOutOfBounds", &InterpTable3D::ThrowOutOfBounds);

  obj.def("__call__",
          py::overload_cast<double, double, double>(&InterpTable3D::interp, py::const_),
          py::is_operator());

  obj.def("__call__",
          [](std::shared_ptr<InterpTable3D>& self,
             const GenericFunction<-1, 1>& x,
             const GenericFunction<-1, 1>& y,
             const GenericFunction<-1, 1>& z) {
            return GenericFunction<-1, 1>(InterpFunction3D(self).eval(stack(x, y, z)));
          });

  obj.def("__call__",
          [](std::shared_ptr<InterpTable3D>& self,
             const Segment<-1, 1, -1>& x,
             const Segment<-1, 1, -1>& y,
             const Segment<-1, 1, -1>& z) {
            return GenericFunction<-1, 1>(InterpFunction3D(self).eval(stack(x, y, z)));
          });

  obj.def("__call__", [](std::shared_ptr<InterpTable3D>& self, const Segment<-1, 3, -1>& xyz) {
    return GenericFunction<-1, 1>(InterpFunction3D(self).eval(xyz));
  });

  obj.def("__call__", [](std::shared_ptr<InterpTable3D>& self, const GenericFunction<-1, -1>& xyz) {
    return GenericFunction<-1, 1>(InterpFunction3D(self).eval(xyz));
  });

  obj.def("sf", [](std::shared_ptr<InterpTable3D>& self) {
    return GenericFunction<-1, 1>(InterpFunction3D(self));
  });
  obj.def("vf", [](std::shared_ptr<InterpTable3D>& self) {
    return GenericFunction<-1, -1>(InterpFunction3D(self));
  });

  m.def("InterpTable3DSpeedTest",
        [](const GenericFunction<-1, 1>& tabf,
           double xl,
           double xu,
           double yl,
           double yu,
           double zl,
           double zu,
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

          if (lin) {
            xsamps.setLinSpaced(xl, xu);
            ysamps.setLinSpaced(yl, yu);
            zsamps.setLinSpaced(zl, zu);
          }

          Eigen::VectorXd xyz(3);
          Vector1<double> f;
          f.setZero();

          Utils::Timer Runtimer;
          Runtimer.start();

          double tmp = 0;
          for (int i = 0; i < nsamps; i++) {

            xyz[0] = xsamps[i];
            xyz[1] = ysamps[i];
            xyz[2] = zsamps[i];

            tabf.compute(xyz, f);
            tmp += f[0] / double(i + 3);

            f.setZero();
          }
          Runtimer.stop();
          double tseconds = double(Runtimer.count<std::chrono::microseconds>()) / 1000000;
          fmt::print("Total Time: {0:} ms \n", tseconds * 1000);

          return tmp;
        });
}
