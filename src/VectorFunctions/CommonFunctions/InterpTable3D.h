#pragma once
#include "VectorFunction.h"
#include <unsupported/Eigen/CXX11/Tensor>
#include <pybind11/eigen/tensor.h>
namespace ASSET {

	struct InterpTable3D {

		enum class InterpType {
			cubic_interp,
			linear_interp
		};

		Eigen::VectorXd xs;
		Eigen::VectorXd ys;
		Eigen::VectorXd zs;

        // numpy meshgrid ij format (x,y,z)

        Eigen::Tensor<double, 3> fs;
        Eigen::Tensor<double, 3> fs_dx;
        Eigen::Tensor<double, 3> fs_dy;
        Eigen::Tensor<double, 3> fs_dz;

        Eigen::Tensor<double, 3> fs_dxdy;
        Eigen::Tensor<double, 3> fs_dxdz;
        Eigen::Tensor<double, 3> fs_dydz;

        Eigen::Tensor<double, 3> fs_dxdydz;

        Eigen::Tensor<Eigen::Array<double,8,1>, 3> all_dat;


		Eigen::MatrixXd Cmat;


        InterpType interp_kind = InterpType::linear_interp;

		bool xeven = true;
		bool yeven = true;
		bool zeven = true;

		int xsize;
		double xtotal;
		int ysize;
		double ytotal;
		int zsize;
		double ztotal;

        bool WarnOutOfBounds = true;
        bool ThrowOutOfBounds = false;

        InterpTable3D() {}

        InterpTable3D(const Eigen::VectorXd& Xs, const Eigen::VectorXd& Ys, const Eigen::VectorXd& Zs,
            const Eigen::Tensor<double, 3>& Fs, std::string kind,std::string indexing) {


            this->xs = Xs;
            this->ys = Ys;
            this->zs = Zs;
            this->fs = Fs;

            const auto& d = fs.dimensions();
            
            for (int i = 0; i < fs.dimension(0); i++) {
                //std::cout << this->fs.chip(i,0) << std::endl << std::endl;

            }


            /// <summary>
            /// ///////////////////////////////////////////
            /// </summary>
            if (kind == "cubic" || kind == "Cubic") {
                this->interp_kind = InterpType::cubic_interp;
            }
            else if (kind == "linear" || kind == "Linear") {
                this->interp_kind = InterpType::linear_interp;
            }
            else {
                throw std::invalid_argument("Unrecognized interpolation type");
            }


            xsize = xs.size();
            ysize = ys.size();
            zsize = zs.size();

            if (xsize < 5) {
                throw std::invalid_argument("X coordinates must be larger than 4");
            }
            if (ysize < 5) {
                throw std::invalid_argument("Y  coordinates must be larger than 4");
            }
            if (zsize < 5) {
                throw std::invalid_argument("Z  coordinates must be larger than 4");
            }




            /*
            if (xsize != zs.cols()) {
                throw std::invalid_argument("X coordinates must match cols in Z matrix");
            }
            if (ysize != zs.rows()) {
                throw std::invalid_argument("Y coordinates must match rows in Z matrix");
            }

            
            */


            for (int i = 0; i < xs.size() - 1; i++) {
                if (xs[i + 1] < xs[i]) {
                    throw std::invalid_argument("X Coordinates must be in ascending order");
                }
            }
            for (int i = 0; i < ys.size() - 1; i++) {
                if (ys[i + 1] < ys[i]) {
                    throw std::invalid_argument("Y Coordinates must be in ascending order");
                }
            }
            for (int i = 0; i < zs.size() - 1; i++) {
                if (zs[i + 1] < zs[i]) {
                    throw std::invalid_argument("Z Coordinates must be in ascending order");
                }
            }

            xtotal = xs[xsize - 1] - xs[0];
            ytotal = ys[ysize - 1] - ys[0];
            ztotal = zs[ysize - 1] - zs[0];


            Eigen::VectorXd testx;
            testx.setLinSpaced(xs.size(), xs[0], xs[xs.size() - 1]);
            Eigen::VectorXd testy;
            testy.setLinSpaced(ys.size(), ys[0], ys[ys.size() - 1]);
            Eigen::VectorXd testz;
            testz.setLinSpaced(zs.size(), zs[0], zs[zs.size() - 1]);

            double xerr = (xs - testx).lpNorm<Eigen::Infinity>();
            double yerr = (ys - testy).lpNorm<Eigen::Infinity>();
            double zerr = (zs - testz).lpNorm<Eigen::Infinity>();

            if (xerr > abs(xtotal) * 1.0e-12) {
                this->xeven = false;
            }
            if (yerr > abs(ytotal) * 1.0e-12) {
                this->yeven = false;
            }
            if (zerr > abs(ztotal) * 1.0e-12) {
                this->zeven = false;
            }


        }



		void fill_Cmat() {
            const int coeffs[64][64] =
            { {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
             {-3, 3, 0, 0, 0, 0, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
             {2, -2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
             {-3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
             {9, -9, -9, 9, 0, 0, 0, 0, 6, 3, -6, -3, 0, 0, 0, 0, 6, -6, 3, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
             {-6, 6, 6, -6, 0, 0, 0, 0, -3, -3, 3, 3, 0, 0, 0, 0, -4, 4, -2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -2, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
             {2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
             {-6, 6, 6, -6, 0, 0, 0, 0, -4, -2, 4, 2, 0, 0, 0, 0, -3, 3, -3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -1, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
             {4, -4, -4, 4, 0, 0, 0, 0, 2, 2, -2, -2, 0, 0, 0, 0, 2, -2, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, 0, 0, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, -9, -9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 3, -6, -3, 0, 0, 0, 0, 6, -6, 3, -3, 0, 0, 0, 0, 4, 2, 2, 1, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -6, 6, 6, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, -3, 3, 3, 0, 0, 0, 0, -4, 4, -2, 2, 0, 0, 0, 0, -2, -2, -1, -1, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -6, 6, 6, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, -2, 4, 2, 0, 0, 0, 0, -3, 3, -3, 3, 0, 0, 0, 0, -2, -1, -2, -1, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, -4, -4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, -2, -2, 0, 0, 0, 0, 2, -2, 2, -2, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0},
             {-3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
             {9, -9, 0, 0, -9, 9, 0, 0, 6, 3, 0, 0, -6, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, -6, 0, 0, 3, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 2, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
             {-6, 6, 0, 0, 6, -6, 0, 0, -3, -3, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 4, 0, 0, -2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -2, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, -1, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, -9, 0, 0, -9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 3, 0, 0, -6, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, -6, 0, 0, 3, -3, 0, 0, 4, 2, 0, 0, 2, 1, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -6, 6, 0, 0, 6, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, -3, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 4, 0, 0, -2, 2, 0, 0, -2, -2, 0, 0, -1, -1, 0, 0},
             {9, 0, -9, 0, -9, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 3, 0, -6, 0, -3, 0, 6, 0, -6, 0, 3, 0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 2, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 9, 0, -9, 0, -9, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 3, 0, -6, 0, -3, 0, 6, 0, -6, 0, 3, 0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 2, 0, 2, 0, 1, 0},
             {-27, 27, 27, -27, 27, -27, -27, 27, -18, -9, 18, 9, 18, 9, -18, -9, -18, 18, -9, 9, 18, -18, 9, -9, -18, 18, 18, -18, -9, 9, 9, -9, -12, -6, -6, -3, 12, 6, 6, 3, -12, -6, 12, 6, -6, -3, 6, 3, -12, 12, -6, 6, -6, 6, -3, 3, -8, -4, -4, -2, -4, -2, -2, -1},
             {18, -18, -18, 18, -18, 18, 18, -18, 9, 9, -9, -9, -9, -9, 9, 9, 12, -12, 6, -6, -12, 12, -6, 6, 12, -12, -12, 12, 6, -6, -6, 6, 6, 6, 3, 3, -6, -6, -3, -3, 6, 6, -6, -6, 3, 3, -3, -3, 8, -8, 4, -4, 4, -4, 2, -2, 4, 4, 2, 2, 2, 2, 1, 1},
             {-6, 0, 6, 0, 6, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 0, -3, 0, 3, 0, 3, 0, -4, 0, 4, 0, -2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -2, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, -6, 0, 6, 0, 6, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 0, -3, 0, 3, 0, 3, 0, -4, 0, 4, 0, -2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -2, 0, -1, 0, -1, 0},
             {18, -18, -18, 18, -18, 18, 18, -18, 12, 6, -12, -6, -12, -6, 12, 6, 9, -9, 9, -9, -9, 9, -9, 9, 12, -12, -12, 12, 6, -6, -6, 6, 6, 3, 6, 3, -6, -3, -6, -3, 8, 4, -8, -4, 4, 2, -4, -2, 6, -6, 6, -6, 3, -3, 3, -3, 4, 2, 4, 2, 2, 1, 2, 1},
             {-12, 12, 12, -12, 12, -12, -12, 12, -6, -6, 6, 6, 6, 6, -6, -6, -6, 6, -6, 6, 6, -6, 6, -6, -8, 8, 8, -8, -4, 4, 4, -4, -3, -3, -3, -3, 3, 3, 3, 3, -4, -4, 4, 4, -2, -2, 2, 2, -4, 4, -4, 4, -2, 2, -2, 2, -2, -2, -2, -2, -1, -1, -1, -1},
             {2, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
             {-6, 6, 0, 0, 6, -6, 0, 0, -4, -2, 0, 0, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -1, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
             {4, -4, 0, 0, -4, 4, 0, 0, 2, 2, 0, 0, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -6, 6, 0, 0, 6, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, -2, 0, 0, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0, -2, -1, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, -4, 0, 0, -4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0},
             {-6, 0, 6, 0, 6, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, -2, 0, 4, 0, 2, 0, -3, 0, 3, 0, -3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -1, 0, -2, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, -6, 0, 6, 0, 6, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, -2, 0, 4, 0, 2, 0, -3, 0, 3, 0, -3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -1, 0, -2, 0, -1, 0},
             {18, -18, -18, 18, -18, 18, 18, -18, 12, 6, -12, -6, -12, -6, 12, 6, 12, -12, 6, -6, -12, 12, -6, 6, 9, -9, -9, 9, 9, -9, -9, 9, 8, 4, 4, 2, -8, -4, -4, -2, 6, 3, -6, -3, 6, 3, -6, -3, 6, -6, 3, -3, 6, -6, 3, -3, 4, 2, 2, 1, 4, 2, 2, 1},
             {-12, 12, 12, -12, 12, -12, -12, 12, -6, -6, 6, 6, 6, 6, -6, -6, -8, 8, -4, 4, 8, -8, 4, -4, -6, 6, 6, -6, -6, 6, 6, -6, -4, -4, -2, -2, 4, 4, 2, 2, -3, -3, 3, 3, -3, -3, 3, 3, -4, 4, -2, 2, -4, 4, -2, 2, -2, -2, -1, -1, -2, -2, -1, -1},
             {4, 0, -4, 0, -4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, -2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 4, 0, -4, 0, -4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, -2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0},
             {-12, 12, 12, -12, 12, -12, -12, 12, -8, -4, 8, 4, 8, 4, -8, -4, -6, 6, -6, 6, 6, -6, 6, -6, -6, 6, 6, -6, -6, 6, 6, -6, -4, -2, -4, -2, 4, 2, 4, 2, -4, -2, 4, 2, -4, -2, 4, 2, -3, 3, -3, 3, -3, 3, -3, 3, -2, -1, -2, -1, -2, -1, -2, -1},
             {8, -8, -8, 8, -8, 8, 8, -8, 4, 4, -4, -4, -4, -4, 4, 4, 4, -4, 4, -4, -4, 4, -4, 4, 4, -4, -4, 4, 4, -4, -4, 4, 2, 2, 2, 2, -2, -2, -2, -2, 2, 2, -2, -2, 2, 2, -2, -2, 2, -2, 2, -2, 2, -2, 2, -2, 1, 1, 1, 1, 1, 1, 1, 1} };

            for (int i = 0; i < 64; i++)
                for (int j = 0; j < 64; j++)
                    this->Cmat(i, j) = coeffs[i][j];



		}


        void calc_derivs() {

            fs_dx.resize(fs.dimension(0), fs.dimension(1), fs.dimension(2));
            fs_dy.resize(fs.dimension(0), fs.dimension(1), fs.dimension(2));
            fs_dz.resize(fs.dimension(0), fs.dimension(1), fs.dimension(2));
            fs_dxdy.resize(fs.dimension(0), fs.dimension(1), fs.dimension(2));
            fs_dxdz.resize(fs.dimension(0), fs.dimension(1), fs.dimension(2));
            fs_dydz.resize(fs.dimension(0), fs.dimension(1), fs.dimension(2));
            fs_dxdydz.resize(fs.dimension(0), fs.dimension(1), fs.dimension(2));



 
            auto fdiffimpl = [&](int dir,bool even, const auto& ts, const auto & src,  auto & dest) {

                Eigen::Matrix<double, 5, 5> stens;
                stens.row(0).setOnes();
                Eigen::Matrix<double, 5, 1> rhs;
                rhs << 0, 1, 0, 0, 0;
                Eigen::Matrix<double, 5, 1> times;
                Eigen::Matrix<double, 5, 1> coeffs;
                int tsize = ts.size();

                bool hitcent = false;
                for (int i = 0; i < tsize; i++) {
                    int start = 0;
                    bool recalc = true;
                    if (i + 2 <= tsize - 1 && i - 2 >= 0) {
                        // central difference
                        if (even && hitcent) {
                            recalc = false;
                        }
                        hitcent = true;
                        start = i - 2;
                    }
                    else if (i < tsize - 1 - i) {
                        // forward difference
                        start = 0;
                    }
                    else {
                        // backward difference
                        start = tsize - 5;
                    }
                    int stepdir = (i < tsize - 1) ? 1 : -1;
                    double ti = ts[i];
                    double tstep = std::abs(ts[i + stepdir] - ti);
                    if (recalc) {
                        times = ts.segment(start, 5);
                        times -= Eigen::Matrix<double, 5, 1>::Constant(ti);
                        times /= tstep;
                        stens.row(1) = times.transpose();
                        stens.row(2) = stens.row(1).cwiseProduct(times.transpose());
                        stens.row(3) = stens.row(2).cwiseProduct(times.transpose());
                        stens.row(4) = stens.row(3).cwiseProduct(times.transpose());
                        coeffs = stens.inverse() * rhs;
                    }
                    dest.chip(i, dir) 
                        = src.chip(start, dir) * (coeffs[0] / tstep)
                        + src.chip(start + 1, dir) * (coeffs[1] / tstep)
                        + src.chip(start + 2, dir) * (coeffs[2] / tstep)
                        + src.chip(start + 3, dir) * (coeffs[3] / tstep)
                        + src.chip(start + 4, dir) * (coeffs[4] / tstep);

                }
            };


            fdiffimpl(0, this->xeven, this->xs, this->fs, this->fs_dx);
            fdiffimpl(1, this->yeven, this->ys, this->fs, this->fs_dy);
            fdiffimpl(2, this->zeven, this->zs, this->fs, this->fs_dz);
    
            fdiffimpl(1, this->yeven, this->ys, this->fs_dx, this->fs_dxdy);
            fdiffimpl(2, this->zeven, this->zs, this->fs_dx, this->fs_dxdz);
            fdiffimpl(2, this->zeven, this->zs, this->fs_dy, this->fs_dydz);

            fdiffimpl(2, this->zeven, this->zs, this->fs_dxdy, this->fs_dxdydz);

        }



        int find_elem(const Eigen::VectorXd& vs, double v) const {
            int center = int(vs.size() / 2);
            int shift = (vs[center] > v) ? 0 : center;
            auto it = std::upper_bound(vs.begin() + shift, vs.end(), v);
            int elem = int(it - vs.begin()) - 1;
            return elem;
        }

        std::tuple<int, int,int> get_xyzelems(double x, double y,double z) const {
            int xelem, yelem,zelem;

            if (this->xeven) {
                double xlocal = x - this->xs[0];
                double xstep = this->xs[1] - this->xs[0];
                xelem = std::min(int(xlocal / xstep), this->xsize - 2);
            }
            else {
                xelem = this->find_elem(this->xs, x);
            }

            if (this->yeven) {
                double ylocal = y - this->ys[0];
                double ystep = this->ys[1] - this->ys[0];
                yelem = std::min(int(ylocal / ystep), this->ysize - 2);
            }
            else {
                yelem = this->find_elem(this->ys, y);
            }

            if (this->zeven) {
                double zlocal = z - this->zs[0];
                double zstep = this->zs[1] - this->zs[0];
                zelem = std::min(int(zlocal / zstep), this->zsize - 2);
            }
            else {
                zelem = this->find_elem(this->zs, z);
            }


            xelem = std::min(xelem, this->xsize - 2);
            yelem = std::min(yelem, this->ysize - 2);
            zelem = std::min(zelem, this->zsize - 2);

            xelem = std::max(xelem, 0);
            yelem = std::max(yelem, 0);
            zelem = std::max(zelem, 0);

            return std::tuple{ xelem,yelem,zelem };
        }

        Eigen::Matrix<double, 64, 1> get_alphavec(int xelem, int yelem, int zelem) const {

            double xstep = xs[xelem + 1] - xs[xelem];
            double ystep = ys[yelem + 1] - ys[yelem];
            double zstep = zs[zelem + 1] - zs[yelem];

            Eigen::Matrix<double, 64, 1> bvec;
            Eigen::Matrix<double, 64, 1> alphavec;


            auto fillop = [&](auto start, const auto & src) {
                for (int i = 0; i < 8; i++) {
                    bvec[start]   = src(xelem  , yelem, zelem);
                    bvec[start+1] = src(xelem+1, yelem, zelem);
                    bvec[start+2] = src(xelem+1, yelem+1, zelem);
                    bvec[start+3] = src(xelem  , yelem+1, zelem);
                    bvec[start+4] = src(xelem, yelem, zelem);
                    bvec[start+5] = src(xelem + 1, yelem, zelem);
                    bvec[start+6] = src(xelem + 1, yelem + 1, zelem);
                    bvec[start+7] = src(xelem, yelem + 1, zelem);
                }
            };
            
            fillop(0, this->fs);
            fillop(8 , this->fs_dx);
            fillop(16, this->fs_dy);
            fillop(24, this->fs_dz);
            fillop(32, this->fs_dxdy);
            fillop(40, this->fs_dxdz);
            fillop(48, this->fs_dydz);
            fillop(56, this->fs_dxdydz);


            bvec.segment(8, 8)  *= (xstep);
            bvec.segment(16, 8) *= (ystep);
            bvec.segment(24, 8) *= (zstep);
            bvec.segment(32, 8) *= (xstep * ystep);
            bvec.segment(40, 8) *= (xstep * zstep);
            bvec.segment(48, 8) *= (ystep * zstep);
            bvec.segment(56, 8) *= (xstep * ystep * zstep);


            alphavec = this->Cmat * bvec;

            return alphavec;
        }


        void interp_impl(double x, double y, double z,
            int deriv,
            double& fval,
            Eigen::Vector3<double>& dfxyz,
            Eigen::Matrix3<double>& d2fxyz) const {

            if (WarnOutOfBounds || ThrowOutOfBounds) {
                double xeps = std::numeric_limits<double>::epsilon() * xtotal;
                if (x<(xs[0] - xeps) || x>(xs[xs.size() - 1] + xeps)) {
                    fmt::print("{0}\n", x);
                    fmt::print(fmt::fg(fmt::color::red),
                        "WARNING: x coordinate falls outside of InterpTable2D range. Data is being extrapolated!!\n");
                    if (ThrowOutOfBounds) {
                        throw std::invalid_argument("");
                    }
                }
                double yeps = std::numeric_limits<double>::epsilon() * ytotal;
                if (y<(ys[0] - yeps) || y>(ys[ys.size() - 1]) + yeps) {
                    fmt::print(fmt::fg(fmt::color::red),
                        "WARNING: y coordinate falls outside of InterpTable2D range. Data is being extrapolated!!\n");
                    if (ThrowOutOfBounds) {
                        throw std::invalid_argument("");
                    }
                }
                double zeps = std::numeric_limits<double>::epsilon() * ztotal;
                if (z<(zs[0] - zeps) || z>(zs[zs.size() - 1]) + zeps) {
                    fmt::print(fmt::fg(fmt::color::red),
                        "WARNING: z coordinate falls outside of InterpTable2D range. Data is being extrapolated!!\n");
                    if (ThrowOutOfBounds) {
                        throw std::invalid_argument("");
                    }
                }

            }


            auto [xelem, yelem, zelem] = get_xyzelems(x, y, z);

            double xstep = xs[xelem + 1] - xs[xelem];
            double ystep = ys[yelem + 1] - ys[yelem];
            double zstep = zs[zelem + 1] - zs[zelem];

            double xf = (x - xs[xelem]) / xstep;
            double yf = (y - ys[yelem]) / ystep;
            double zf = (z - zs[zelem]) / zstep;

            if (this->interp_kind == InterpType::cubic_interp) {
                Eigen::Matrix<double, 64, 1> alphavec = this->get_alphavec(xelem,yelem,zelem);

                double xf2 = xf * xf;
                double xf3 = xf2 * xf;

                double yf2 = yf * yf;
                double yf3 = yf2 * yf;

                double zf2 = zf * zf;
                double zf3 = zf2 * zf;

                Eigen::Vector4d yfs{ 1,yf,yf2,yf3 };
                Eigen::Vector4d zfs{ 1,zf,zf2,zf3 };


                int start = 0;
                fval = 0;

                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        fval += (yfs[j] * zfs[i]) * (alphavec[start] + xf * alphavec[start + 1] + xf2 * alphavec[start + 2] + xf3 * alphavec[start + 3]);
                        start += 4;
                    }
                }

                if (deriv > 0) {

                    
                    dfxyz[0] = 0;
                    dfxyz[1] = 0;
                    dfxyz[2] = 0;

                    int start = 0;
                    for (int i = 0; i < 4; i++) {
                        for (int j = 0; j < 4; j++) {
                            dfxyz[0] += (yfs[j] * zfs[i]) * (alphavec[start + 1] + 2*xf * alphavec[start + 2] + 3*xf2 * alphavec[start + 3]);
                            if (j > 0) {
                                dfxyz[1] += (j*yfs[j-1] * zfs[i]) * (alphavec[start] + xf * alphavec[start + 1] + xf2 * alphavec[start + 2] + xf3 * alphavec[start + 3]);
                            }
                            if (i > 0) {
                                dfxyz[2] += (yfs[j] * i*zfs[i-1]) * (alphavec[start] + xf * alphavec[start + 1] + xf2 * alphavec[start + 2] + xf3 * alphavec[start + 3]);
                            }
                            start += 4;
                        }
                    }


                   

                    if (deriv > 1) {
                        d2fxyz.setZero();
                        
                        
                        /// second partial wrt. x /////////
                     
                        start = 0;
                        for (int i = 0; i < 4; i++) {
                            for (int j = 0; j < 4; j++) {
                                d2fxyz(0,0) += (yfs[j] * zfs[i]) * (2 * alphavec[start + 2] + 6 * xf * alphavec[start + 3]);
                                if (j > 0) {
                                    d2fxyz(1, 0) += (j * yfs[j - 1] * zfs[i]) * (alphavec[start + 1] + 2 * xf * alphavec[start + 2] + 3 * xf2 * alphavec[start + 3]);
                                }
                                if (i > 0) {
                                    d2fxyz(2, 0) += (yfs[j] * i * zfs[i - 1]) * (alphavec[start + 1] + 2 * xf * alphavec[start + 2] + 3 * xf2 * alphavec[start + 3]);
                                }
                                start += 4;
                            }
                        }

                        /// second partial wrt. y /////////
                        start = 0;
                        for (int i = 0; i < 4; i++) {
                            for (int j = 0; j < 4; j++) {
                                if (j > 0) {
                                    d2fxyz(0, 1) += (j*yfs[j-1] * zfs[i]) * (alphavec[start + 1] + 2 * xf * alphavec[start + 2] + 3 * xf2 * alphavec[start + 3]);
                                }
                                if (j > 1) {
                                    d2fxyz(1, 1) += (j*(j-1) * yfs[j - 2] * zfs[i]) * (alphavec[start] + xf * alphavec[start + 1] + xf2 * alphavec[start + 2] + xf3 * alphavec[start + 3]);
                                }
                                if (i > 0) {
                                    if (j > 0) {
                                        d2fxyz(2, 1) += (j * yfs[j - 1] * i * zfs[i - 1]) * (alphavec[start] + xf * alphavec[start + 1] + xf2 * alphavec[start + 2] + xf3 * alphavec[start + 3]);
                                    }
                                }
                                start += 4;
                            }
                        }


                        /// second partial wrt. z /////////

                        


                    }
                }

            }
            else {
                double c000 = this->fs(xelem,yelem, zelem);
                double c100 = this->fs(xelem + 1,yelem, zelem);

                double c001 = this->fs(xelem, yelem, zelem + 1);
                double c101 = this->fs(xelem + 1, yelem, zelem + 1);

                double c010 = this->fs(xelem, yelem + 1, zelem);
                double c110 = this->fs(xelem + 1, yelem + 1, zelem);

                double c011 = this->fs(xelem, yelem + 1, zelem + 1);
                double c111 = this->fs(xelem + 1, yelem + 1, zelem + 1);

                double x0 = this->xs[xelem];
                double x1 = this->xs[xelem + 1];

                double y0 = this->ys[yelem];
                double y1 = this->ys[yelem + 1];

                double z0 = this->zs[zelem];
                double z1 = this->zs[zelem + 1];

                double scale = -1.0 / (xstep * ystep * zstep);


                double a0 = (-c000 * x1 * y1 * z1 + c001 * x1 * y1 * z0 + c010 * x1 * y0 * z1 - c011 * x1 * y0 * z0
                    + c100 * x0 * y1 * z1 - c101 * x0 * y1 * z0 - c110 * x0 * y0 * z1 + c111 * x0 * y0 * z0) * scale;

                double a1 = (c000 * y1 * z1 - c001 * y1 * z0 - c010 * y0 * z1 + c011 * y0 * z0
                    - c100 * y1 * z1 + c101 * y1 * z0 + c110 * y0 * z1 - c111 * y0 * z0) * scale;

                double a2 = (c000 * x1 * z1 - c001 * x1 * z0 - c010 * x1 * z1 + c011 * x1 * z0
                    - c100 * x0 * z1 + c101 * x0 * z0 + c110 * x0 * z1 - c111 * x0 * z0) * scale;

                double a3 = (c000 * x1 * y1 - c001 * x1 * y1 - c010 * x1 * y0 + c011 * x1 * y0
                    - c100 * x0 * y1 + c101 * x0 * y1 + c110 * x0 * y0 - c111 * x0 * y0) * scale;


                double a4 = (-c000 * z1 + c001 * z0 + c010 * z1 - c011 * z0 + c100 * z1 - c101 * z0 - c110 * z1 + c111 * z0) * scale;
                double a5 = (-c000 * y1 + c001 * y1 + c010 * y0 - c011 * y0 + c100 * y1 - c101 * y1 - c110 * y0 + c111 * y0) * scale;
                double a6 = (-c000 * x1 + c001 * x1 + c010 * x1 - c011 * x1 + c100 * x0 - c101 * x0 - c110 * x0 + c111 * x0) * scale;

                double a7 = (c000 - c001 - c010 + c011 - c100 + c101 + c110 - c111) * scale;

                fval = a0 + a1 * x + a2 * y + a3 * z + a4 * x * y + a5 * x * z + a6 * y * z + a7 * x * y * z;

                if (deriv > 0) {
                    dfxyz[0] = a1 + a4 * y + a5 * z + a7 * y * z;
                    dfxyz[1] = a2 + a4 * x + a6 * z + a7 * x * z;
                    dfxyz[2] = a3 + a5 * x + a6 * y + a7 * x * y;

                    if (deriv > 1) {
                        d2fxyz.setZero();
                        d2fxyz(1, 0) = a4 + a7 * z;
                        d2fxyz(2, 0) = a5 + a7 * y;
                        d2fxyz(2, 1) = a6 + a7 * x;
                        d2fxyz(0, 1) = d2fxyz(1, 0);
                        d2fxyz(0, 2) = d2fxyz(2, 0);
                        d2fxyz(1, 2) = d2fxyz(2, 1);

                    }
                }


            }
        }



        double interp(double x, double y,double z) const {

            double f;
            Eigen::Vector3<double> dfxyz;
            Eigen::Matrix3<double> d2fxyz;
            interp_impl(x,y,z, 0, f, dfxyz, d2fxyz);

            return f;

        }
	};

	static void InterpTable3DBuild(py::module& m) {

        auto obj = py::class_<InterpTable3D, std::shared_ptr<InterpTable3D>>(m, "InterpTable3D");

        obj.def(py::init<const Eigen::VectorXd& , const Eigen::VectorXd& , const Eigen::VectorXd& ,
            const Eigen::Tensor<double, 3>& , std::string , std::string  >(),
            py::arg("xs"), py::arg("ys"), py::arg("zs"), py::arg("fs"), 
            py::arg("kind") = std::string("cubic"), py::arg("indexing") = std::string("ij"));


        obj.def("interp", py::overload_cast<double, double,double>(&InterpTable3D::interp, py::const_));
        obj.def("__call__", py::overload_cast<double, double,double>(&InterpTable3D::interp, py::const_), py::is_operator());


	}

}