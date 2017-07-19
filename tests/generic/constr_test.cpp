//
// this example is from Matlab's help page
// https://www.mathworks.com/help/optim/ug/fminunc.html
//
// f(x) = (x_1 - 6)^2 + (x_2 - 7)^2
// g(x) = -3*x_1 - 2*x_2 + 6 <= 0
// 
// solution is: (6,7)
//
// g++-mp-7 -O2 -Wall -std=c++11 -I/opt/local/include constr_test.cpp -o constr.test -L/opt/local/lib -loptim -framework Accelerate
//

#include "optim/optim.hpp"

double opt_objfn(const arma::vec& vals_inp, arma::vec* grad, void* opt_data)
{
    double x_1 = vals_inp(0);
    double x_2 = vals_inp(1);

    double obj_val = std::pow(x_1 - 6.0,2) + std::pow(x_2 - 7.0,2);
    //
    if (grad) {
        (*grad)(0) = 2.0*(x_1 - 6.0);
        (*grad)(1) = 2.0*(x_2 - 7.0);
    }
    //
    return obj_val;
}

double constr_fn(const arma::vec& vals_inp, arma::vec* grad, void* opt_data)
{
    double x_1 = vals_inp(0);
    double x_2 = vals_inp(1);

    double constr_val = -3*x_1 - 2*x_2 + 6.0;
    //
    if (grad) {
        (*grad)(0) = -3.0;
        (*grad)(1) = -2.0;
    }
    //
    return constr_val;
}

int main()
{
    //
    arma::vec lower_bounds(2), upper_bounds(2);
    lower_bounds.fill(0.0);
    upper_bounds.fill(10.0);

    arma::vec x = arma::ones(2,1);

    bool success = optim::generic_constr_optim(x,lower_bounds,upper_bounds,opt_objfn,NULL,constr_fn,NULL);

    arma::cout << x << arma::endl;

    return 0;
}
