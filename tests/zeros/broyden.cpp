
//
// g++-mp-7 -O2 -Wall -std=c++11 -I./../../include broyden.cpp -o broyden.test -L./../.. -loptim -framework Accelerate
//

#include "optim.hpp"
#include "./../test_fns/test_fns.hpp"

// this example is from Matlab's help page
// https://www.mathworks.com/help/optim/ug/fsolve.html
//
// F = [2*x_1 - x_2   - exp(-x_1);
//      -x_1  + 2*x_2 - exp(-x_2)]
// 
// solution is: (0.5671,0.5671)
//

int main()
{
    //
    // test 1

    arma::vec x_1 = arma::zeros(2,1);

    bool success_1 = optim::broyden_df(x_1,zeros_test_objfn_1,nullptr);

    if (success_1) {
        std::cout << "broyden_df: test_1 completed successfully." << std::endl;
    } else {
        std::cout << "broyden_df: test_1 completed unsuccessfully." << std::endl;
    }

    arma::cout << "broyden_df: solution to test_1:\n" << x_1 << arma::endl;

    //
    // test 2

    arma::vec x_2 = arma::zeros(2,1);

    bool success_2 = optim::broyden_df(x_2,zeros_test_objfn_2,nullptr);

    if (success_2) {
        std::cout << "broyden_df: test_2 completed successfully." << std::endl;
    } else {
        std::cout << "broyden_df: test_2 completed unsuccessfully." << std::endl;
    }

    arma::cout << "broyden_df: solution to test_2:\n" << x_2 << arma::endl;

    //
    // coverage tests

    optim::algo_settings settings;

    optim::broyden_df(x_1,zeros_test_objfn_1,nullptr);
    optim::broyden_df(x_1,zeros_test_objfn_1,nullptr,settings);

    //
    // test 1

    x_1 = arma::zeros(2,1);

    success_1 = optim::broyden_df(x_1,zeros_test_objfn_1,nullptr,zeros_test_jacob_1,nullptr);

    if (success_1) {
        std::cout << "broyden_df w jacobian: test_1 completed successfully." << std::endl;
    } else {
        std::cout << "broyden_df w jacobian: test_1 completed unsuccessfully." << std::endl;
    }

    arma::cout << "broyden_df w jacobian: solution to test_1:\n" << x_1 << arma::endl;

    //
    // test 2

    x_2 = arma::zeros(2,1);

    success_2 = optim::broyden_df(x_2,zeros_test_objfn_2,nullptr,zeros_test_jacob_2,nullptr);

    if (success_2) {
        std::cout << "broyden_df w jacobian: test_2 completed successfully." << std::endl;
    } else {
        std::cout << "broyden_df w jacobian: test_2 completed unsuccessfully." << std::endl;
    }

    arma::cout << "broyden_df w jacobian: solution to test_2:\n" << x_2 << arma::endl;

    //
    // coverage tests

    optim::broyden_df(x_1,zeros_test_objfn_1,nullptr,zeros_test_jacob_1,nullptr);
    optim::broyden_df(x_1,zeros_test_objfn_1,nullptr,zeros_test_jacob_1,nullptr,settings);

    //
    // test 1

    x_1 = arma::zeros(2,1);

    success_1 = optim::broyden(x_1,zeros_test_objfn_1,nullptr);

    if (success_1) {
        std::cout << "broyden: test_1 completed successfully." << std::endl;
    } else {
        std::cout << "broyden: test_1 completed unsuccessfully." << std::endl;
    }

    arma::cout << "broyden: solution to test_1:\n" << x_1 << arma::endl;

    //
    // test 2

    x_2 = arma::zeros(2,1);

    success_2 = optim::broyden(x_2,zeros_test_objfn_2,nullptr);

    if (success_2) {
        std::cout << "broyden: test_2 completed successfully." << std::endl;
    } else {
        std::cout << "broyden: test_2 completed unsuccessfully." << std::endl;
    }

    arma::cout << "broyden: solution to test_2:\n" << x_2 << arma::endl;

    //
    // coverage tests

    optim::broyden(x_1,zeros_test_objfn_1,nullptr);
    optim::broyden(x_1,zeros_test_objfn_1,nullptr,settings);

    //
    // test 1

    x_1 = arma::zeros(2,1);

    success_1 = optim::broyden(x_1,zeros_test_objfn_1,nullptr,zeros_test_jacob_1,nullptr);

    if (success_1) {
        std::cout << "broyden w jacobian: test_1 completed successfully." << std::endl;
    } else {
        std::cout << "broyden w jacobian: test_1 completed unsuccessfully." << std::endl;
    }

    arma::cout << "broyden w jacobian: solution to test_1:\n" << x_1 << arma::endl;

    //
    // test 2

    x_2 = arma::zeros(2,1);

    success_2 = optim::broyden(x_2,zeros_test_objfn_2,nullptr,zeros_test_jacob_2,nullptr);

    if (success_2) {
        std::cout << "broyden w jacobian: test_2 completed successfully." << std::endl;
    } else {
        std::cout << "broyden w jacobian: test_2 completed unsuccessfully." << std::endl;
    }

    arma::cout << "broyden w jacobian: solution to test_2:\n" << x_2 << arma::endl;

    //
    // coverage tests

    optim::broyden(x_1,zeros_test_objfn_1,nullptr,zeros_test_jacob_1,nullptr);
    optim::broyden(x_1,zeros_test_objfn_1,nullptr,zeros_test_jacob_1,nullptr,settings);


    return 0;
}
