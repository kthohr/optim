//
// this example is from
// https://en.wikipedia.org/wiki/Test_functions_for_optimization
//
// f(x) = (x_1 + 2*x_2 - 7)^2 + (2*x + y - 5)^2   s.t. -10 <= x_1, x_2 <= 10
// 
// solution is: (1,3)
//
// g++-mp-5 -O2 -Wall -std=c++11 -I/opt/local/include booth_test.cpp -o booth.test -L/opt/local/lib -loptim -framework Accelerate
//

#include "optim/optim.hpp"

double booth_obj(const arma::vec& vals_inp, arma::vec* grad, void* opt_data)
{
    double x_1 = vals_inp(0);
    double x_2 = vals_inp(1);

    double obj_val = std::pow(x_1 + 2*x_2 - 7.0,2) + std::pow(2*x_1 + x_2 - 5.0,2);
    //
    if (grad) {
        (*grad)(0) = 2*(x_1 + 2*x_2 - 7.0) + 2*(2*x_1 + x_2 - 5.0)*2;
        (*grad)(1) = 2*(x_1 + 2*x_2 - 7.0)*2 + 2*(2*x_1 + x_2 - 5.0);
    }
    //
    return obj_val;
}

int main()
{
    //
    arma::vec lower_bounds(2), upper_bounds(2);
    lower_bounds.fill(-10.0);
    upper_bounds.fill(10.0);

    arma::vec x = arma::ones(2,1) + 4.0;

    bool success = optim::generic_optim(x,lower_bounds,upper_bounds,booth_obj,NULL);

    arma::cout << x << arma::endl;

    return 0;
}
