//
// this example is from Matlab's help page
// https://www.mathworks.com/help/optim/ug/fsolve.html
//
// F(x) = [exp(-exp(-(x_1+x_2))) - x_2*(1+x_1^2);
//         x_1*cos(x_2) + x_2*sin(x_1) - 0.5     ]
// 
// solution is: (0.3532,0.6061)
//
// g++-mp-7 -O2 -Wall -std=c++11 -I./../../../include broyden_test_1.cpp -o broyden_1.test -L./../../.. -loptim -framework Accelerate
//

#include "optim.hpp"

arma::vec zero_obj(const arma::vec& vals_inp, void* opt_data)
{
    double x_1 = vals_inp(0);
    double x_2 = vals_inp(1);

    arma::vec ret(2);

    ret(0) = std::exp(-std::exp(-(x_1+x_2))) - x_2*(1 + std::pow(x_1,2));
    ret(1) = x_1*std::cos(x_2) + x_2*std::sin(x_1) - 0.5;
    //
    return ret;
}

int main()
{
    //
    arma::vec x = arma::zeros(2,1);

    //bool success = broyden(x,zero_obj,NULL);
    bool success = optim::broyden_df(x,zero_obj,NULL);

    arma::cout << x << arma::endl;

    return 0;
}
