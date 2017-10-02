//
// BFGS tests
//
// g++-mp-7 -O2 -Wall -std=c++11 -I/opt/local/include lbfgs_test.cpp -o lbfgs.test -L/opt/local/lib -loptim -framework Accelerate
// g++-mp-7 -O2 -Wall -std=c++11 -I./../../include lbfgs.cpp -o lbfgs.test -L./../.. -loptim -framework Accelerate
//

#include "optim.hpp"
#include "./../test_fns/test_fns.hpp"

int main()
{
    //
    // test 1
    arma::vec x_1 = arma::ones(2,1);

    bool success_1 = optim::lbfgs(x_1,unconstr_test_fn_1,nullptr);

    if (success_1) {
        std::cout << "lbfgs: test_1 completed successfully." << std::endl;
    } else {
        std::cout << "lbfgs: test_1 completed unsuccessfully." << std::endl;
    }

    arma::cout << "lbfgs: solution to test_1:\n" << x_1 << arma::endl;

    //
    // test 2

    arma::vec x_2 = arma::zeros(2,1);

    bool success_2 = optim::lbfgs(x_2,unconstr_test_fn_2,nullptr);

    if (success_2) {
        std::cout << "lbfgs: test_2 completed successfully." << std::endl;
    } else {
        std::cout << "lbfgs: test_2 completed unsuccessfully." << std::endl;
    }

    arma::cout << "lbfgs: solution to test_2:\n" << x_2 << arma::endl;

    //
    // test 3
    int test_3_dim = 5;
    arma::vec x_3 = arma::ones(test_3_dim,1);

    bool success_3 = optim::lbfgs(x_3,unconstr_test_fn_3,nullptr);

    if (success_3) {
        std::cout << "lbfgs: test_3 completed successfully." << std::endl;
    } else {
        std::cout << "lbfgs: test_3 completed unsuccessfully." << std::endl;
    }

    arma::cout << "lbfgs: solution to test_3:\n" << x_3 << arma::endl;

    //
    // test 4
    arma::vec x_4 = arma::ones(2,1);

    bool success_4 = optim::lbfgs(x_4,unconstr_test_fn_4,nullptr);

    if (success_4) {
        std::cout << "lbfgs: test_4 completed successfully." << std::endl;
    } else {
        std::cout << "lbfgs: test_4 completed unsuccessfully." << std::endl;
    }

    arma::cout << "lbfgs: solution to test_4:\n" << x_4 << arma::endl;

    //
    // test 5
    arma::vec x_5 = arma::zeros(2,1);

    bool success_5 = optim::lbfgs(x_5,unconstr_test_fn_5,nullptr);

    if (success_5) {
        std::cout << "lbfgs: test_5 completed successfully." << std::endl;
    } else {
        std::cout << "lbfgs: test_5 completed unsuccessfully." << std::endl;
    }

    arma::cout << "lbfgs: solution to test_5:\n" << x_5 << arma::endl;

    //
    // for coverage

    optim::algo_settings settings;

    optim::lbfgs(x_1,unconstr_test_fn_1,nullptr);
    optim::lbfgs(x_1,unconstr_test_fn_1,nullptr,settings);

    settings.vals_bound = true;
    settings.lower_bounds = arma::zeros(2,1) - 4.5;
    settings.upper_bounds = arma::zeros(2,1) + 4.5;

    x_4 = arma::ones(2,1);
    
    success_4 = optim::lbfgs(x_4,unconstr_test_fn_4,nullptr,settings);

    arma::cout << "lbfgs: solution to test_4 with box constraints:\n" << x_4 << arma::endl;

    return 0;
}
