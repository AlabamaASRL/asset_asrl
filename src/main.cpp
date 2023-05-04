#include <signal.h>

#include "Astro/ASSET_Astro.h"
#include "OptimalControl/ASSET_OptimalControl.h"
#include "Solvers/ASSET_Solvers.h"
#include "Utils/ASSET_Utils.h"
#include "VectorFunctions/ASSET_VectorFunctions.h"
#include "pch.h"


using namespace ASSET;
using namespace rubber_types;


////////////////////////////////////////////////////////////////////////////


void SoftwareInfo() {


  int tcount = std::thread::hardware_concurrency();
  int ccount = ASSET::get_core_count();
  int vsize = ASSET::DefaultSuperScalar::SizeAtCompileTime;


  std::string assetversion = std::string(ASSET_VERSIONSTRING);
  std::string osversion = std::string(ASSET_OS) + " " + std::string(ASSET_OSVERSION);

  std::string syscorecount = std::to_string(ccount);
  std::string systhreadcount = std::to_string(tcount);


  std::string compiler =
      std::string(ASSET_COMPILERSTRING) + std::string(" ") + std::string(ASSET_COMPILERVERSION);
  std::string pythonv = std::to_string(PY_MAJOR_VERSION) + "." + std::to_string(PY_MINOR_VERSION);
  std::string vecversion;
  if (vsize == 2)
    vecversion = "SSE - 128 bit - 2 doubles";
  else if (vsize == 4)
    vecversion = "AVX2 - 256 bit - 4 doubles";
  else if (vsize == 8)
    vecversion = "AVX512 - 512 bit - 8 doubles";


  std::string ASSET_STR("         ___    _____   _____    ______  ______ \n"
                        "        /   |  / ___/  / ___/   / ____/ /_  __/    \n"
                        "       / /| |  \\__ \\   \\__ \\   / __/     / /        \n"
                        "      / ___ | ___/ /  ___/ /  / /___    / /      \n"
                        "     /_/  |_|/____/  /____/  /_____/   /_/     \n\n");


  fmt::print(fmt::fg(fmt::color::white), "{0:=^{1}}\n", "", 79);
  fmt::print(fmt::fg(fmt::color::crimson), ASSET_STR);
  fmt::print(fmt::fg(fmt::color::crimson), " Astrodynamics Software and Science Enabling Toolkit\n");
  fmt::print(fmt::fg(fmt::color::white), "{0:=^{1}}\n", "", 79);
  fmt::print(fmt::fg(fmt::color::white), "\nDevelopment funded by NASA under Grant No. 80NSSC19K1643\n\n");


  fmt::print(fmt::fg(fmt::color::royal_blue), " Senior Personnel:\n");
  fmt::print(fmt::fg(fmt::color::white), "  Rohan Sood            rsood@eng.ua.edu                PI\n");
  fmt::print(fmt::fg(fmt::color::white), "  Kathleen Howell       howell@purdue.edu               Co-I\n");
  fmt::print(fmt::fg(fmt::color::white), "  Jeff Stuart           jeffrey.r.stuart@jpl.nasa.gov   Co-I\n");


  fmt::print(fmt::fg(fmt::color::royal_blue), " Student Contributors:\n");
  fmt::print(fmt::fg(fmt::color::white),
             "  James B. Pezent       jbpezent@crimson.ua.edu         Lead Developer\n");
  fmt::print(fmt::fg(fmt::color::white),
             "  Jared D. Sikes        jdsikes1@crimson.ua.edu         Developer\n");
  fmt::print(fmt::fg(fmt::color::white),
             "  William G. Ledbetter  wgledbetter@crimson.ua.edu      Developer\n");
  fmt::print(fmt::fg(fmt::color::white),
             "  Carrie G. Sandel      cgsandel@crimson.ua.edu         Developer\n");


  fmt::print(fmt::fg(fmt::color::white), "{0:=^{1}}\n\n", "", 79);
  fmt::print(fmt::fg(fmt::color::royal_blue), " Software Version     : ");
  fmt::print(fmt::fg(fmt::color::white), assetversion);
  fmt::print("\n");
  fmt::print(fmt::fg(fmt::color::royal_blue), " Python   Version     : ");
  fmt::print(fmt::fg(fmt::color::white), pythonv);
  fmt::print("\n");

  fmt::print(fmt::fg(fmt::color::royal_blue), " System Core Count    : ");
  fmt::print(fmt::fg(fmt::color::white), syscorecount);
  fmt::print("\n");
  fmt::print(fmt::fg(fmt::color::royal_blue), " System Thread Count  : ");
  fmt::print(fmt::fg(fmt::color::white), systhreadcount);
  fmt::print("\n");
  fmt::print(fmt::fg(fmt::color::royal_blue), " Vectorization Mode   : ");
  fmt::print(fmt::fg(fmt::color::white), vecversion);
  fmt::print("\n");
  fmt::print(fmt::fg(fmt::color::royal_blue), " Linear Solver        : ");
  fmt::print(fmt::fg(fmt::color::white), "Intel MKL Pardiso");
  fmt::print("\n");

  fmt::print(fmt::fg(fmt::color::royal_blue), " Compiled With        : ");
  fmt::print(fmt::fg(fmt::color::white), compiler);
  fmt::print("\n");
  fmt::print(fmt::fg(fmt::color::royal_blue), " Compiled On          : ");
  fmt::print(fmt::fg(fmt::color::white), osversion);
  fmt::print("\n");


  fmt::print(fmt::fg(fmt::color::white), "{0:=^{1}}\n\n", "", 79);
  fmt::print(fmt::fg(fmt::color::royal_blue),
             " Copyright/Licensing Notices : See package's .dist.data folder for full text\n");

  fmt::print(fmt::fg(fmt::color::white),
             "  ASSET    :Apache 2.0 | Copyright (c) 2020-present The University of Alabama-Astrodynamics "
             "and Space Research Lab\n");
  fmt::print(fmt::fg(fmt::color::white),
             "  Pybind11 :Modified BSD | Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>, All rights "
             "reserved.\n");
  fmt::print(fmt::fg(fmt::color::white),
             "  Intel MKL:Intel Simplified Software License (Version October 2022) | Copyright (c) 2022 "
             "Intel Corporation \n");
  fmt::print(fmt::fg(fmt::color::white), "  Eigen    :MPL-2.0. | Copyright (c) Eigen Developers \n");
  fmt::print(fmt::fg(fmt::color::white),
             "  fmt      :MIT | Copyright (c) 2012 - present, Victor Zverovich \n");
}

void signal_callback(int sig) {
  fmt::print(
      fmt::fg(fmt::color::red), "Interrupt signal [{0}] received, terminating program.\n\n\n\n\n\n\n\n", sig);
  exit(sig);
}

int main() {
  using std::cin;
  using std::cout;
  using std::endl;


  ASSET::enable_color_console();

  signal(SIGINT, signal_callback);

  SoftwareInfo();

  return 0;
}


PYBIND11_MODULE(asset, m) {

  ASSET::enable_color_console();  // This only does something on windows

  signal(SIGINT, signal_callback);

  m.doc() = "ASSET";  // optional module docstring
  m.def("PyMain", &main);
  m.def("SoftwareInfo", &SoftwareInfo);


  FunctionRegistry reg(m);      // Must be built first
  VectorFunctionBuild(reg, m);  // Must be built second
  SolversBuild(reg, m);         // Builds Third so that PSIOPT shows up better in autocomplete
  OptimalControlBuild(reg, m);
  UtilsBuild(m);
  AstroBuild(reg, m);
}
