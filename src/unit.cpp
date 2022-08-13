#include "VectorFunctions/ASSET_VectorFunctions.h"
#include "VectorFunctions/CommonFunctions/Comparative.h"
#include "pch.h"

using namespace ASSET;
using std::cin;
using std::cout;
using std::endl;

////////////////////////////////////////////////////////////////////////////////

int main() {
  Arguments<1> f1;
  auto f2 = f1 * f1;
  auto f3 = 2 * f1 + 2;

  // ----------------------------------

  ComparativeFunction<Arguments<1>, decltype(f2)> comp1(
      ComparativeFlags::MaxFlag, f1, f2);

  ComparativeFunction<Arguments<1>, decltype(f2), decltype(f3)> comp2(
      ComparativeFlags::MinFlag, f1, f2, f3);

  // ----------------------------------

  Eigen::Matrix<double, 1, 1> inp, out;

  while (true) {
    cout << "Input x value: ";
    cin >> inp[0];

    f1.compute(inp, out);
    cout << "f1 = " << out[0] << endl;
    f2.compute(inp, out);
    cout << "f2 = " << out[0] << endl;
    f3.compute(inp, out);
    cout << "f3 = " << out[0] << endl;

    comp1.compute(inp, out);
    cout << "Max of 'x' and 'x^2' at x = " << inp[0] << " is " << out[0]
         << endl;

    comp2.compute(inp, out);
    cout << "Min of 'x', 'x^2', and '2x+2' at x = " << inp[0] << " is "
         << out[0] << endl;

    cout << endl;
  }
}
