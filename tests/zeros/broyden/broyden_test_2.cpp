//
// this example is from Matlab's help page
// https://www.mathworks.com/help/optim/ug/fsolve.html
//
// F = [2*x_1 - x_2   - exp(-x_1);
//      -x_1  + 2*x_2 - exp(-x_2)]
// 
// solution is: (0.5671,0.5671)
//
// g++-mp-5 -O2 -Wall -std=c++11 -I./../../../include broyden_test_2.cpp -o broyden_2.test -L./../../.. -loptim -framework Accelerate
//


#include "optim.hpp"

arma::vec zero_obj(const arma::vec& vals_inp, void* opt_data)
{
    double x_1 = vals_inp(0);
    double x_2 = vals_inp(1);

    arma::vec ret(2);

    ret(0) = 2*x_1 - x_2 - std::exp(-x_1);
    ret(1) = -x_1 + 2*x_2 - std::exp(-x_2);
    //
    return ret;
}

int main()
{
    //
    //arma::vec x = arma::zeros(2,1) - 5;
    arma::vec x = arma::zeros(2,1);

    bool success = optim::broyden(x,zero_obj,NULL);

    arma::cout << x << arma::endl;

    return 0;
}
