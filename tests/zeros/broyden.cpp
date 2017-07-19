
//
// g++-mp-7 -O2 -Wall -std=c++11 -I./../../include broyden.cpp -o broyden.test -L./../.. -loptim -framework Accelerate
//

#include "optim.hpp"

//
// this example is from Matlab's help page
// https://www.mathworks.com/help/optim/ug/fsolve.html
//
// F(x) = [exp(-exp(-(x_1+x_2))) - x_2*(1+x_1^2);
//         x_1*cos(x_2) + x_2*sin(x_1) - 0.5     ]
// 
// solution is: (0.3532,0.6061)

arma::vec zero_objfn_1(const arma::vec& vals_inp, void* opt_data)
{
    double x_1 = vals_inp(0);
    double x_2 = vals_inp(1);

    arma::vec ret(2);

    ret(0) = std::exp(-std::exp(-(x_1+x_2))) - x_2*(1 + std::pow(x_1,2));
    ret(1) = x_1*std::cos(x_2) + x_2*std::sin(x_1) - 0.5;
    //
    return ret;
}

//
// this example is from Matlab's help page
// https://www.mathworks.com/help/optim/ug/fsolve.html
//
// F = [2*x_1 - x_2   - exp(-x_1);
//      -x_1  + 2*x_2 - exp(-x_2)]
// 
// solution is: (0.5671,0.5671)
//

arma::vec zero_objfn_2(const arma::vec& vals_inp, void* opt_data)
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
    arma::vec x_1 = arma::zeros(2,1);

    bool success_1 = optim::broyden_df(x_1,zero_objfn_1,NULL);

    if (success_1) {
        std::cout << "broyden: test_1 completed successfully." << std::endl;
    } else {
        std::cout << "broyden: test_1 completed unsuccessfully." << std::endl;
    }

    arma::cout << "broyden: solution to test_1:\n" << x_1 << arma::endl;

    //
    // test 2

    arma::vec x_2 = arma::zeros(2,1);

    bool success_2 = optim::broyden_df(x_2,zero_objfn_2,NULL);

    if (success_2) {
        std::cout << "broyden: test_2 completed successfully." << std::endl;
    } else {
        std::cout << "broyden: test_2 completed unsuccessfully." << std::endl;
    }

    arma::cout << "broyden: solution to test_2:\n" << x_2 << arma::endl;

    return 0;
}
