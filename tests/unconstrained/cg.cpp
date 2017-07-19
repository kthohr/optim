//
// CG tests
//
// g++-mp-7 -O2 -Wall -std=c++11 -I/opt/local/include cg_test.cpp -o cg.test -L/opt/local/lib -loptim -framework Accelerate
// g++-mp-7 -O2 -Wall -std=c++11 -I./../../include cg.cpp -o cg.test -L./../.. -loptim -framework Accelerate
//

#include "optim.hpp"
#include "./../test_fns/test_fns.hpp"

int main()
{
    //
    // test 1
    optim::optim_opt_settings opt_params;

    opt_params.iter_max = 2000;
    opt_params.conv_failure_switch = 1;
    opt_params.method_cg = 1;

    arma::vec x_1 = arma::ones(2,1);

    bool success_1 = optim::cg(x_1,unconstr_test_fn_1,NULL,opt_params);

    if (success_1) {
        std::cout << "cg: test_1 completed successfully." << std::endl;
    } else {
        std::cout << "cg: test_1 completed unsuccessfully." << std::endl;
    }

    arma::cout << "cg: solution to test_1:\n" << x_1 << arma::endl;

    //
    // test 2

    arma::vec x_2 = arma::zeros(2,1);

    bool success_2 = optim::cg(x_2,unconstr_test_fn_2,NULL);

    if (success_2) {
        std::cout << "cg: test_2 completed successfully." << std::endl;
    } else {
        std::cout << "cg: test_2 completed unsuccessfully." << std::endl;
    }

    arma::cout << "cg: solution to test_2:\n" << x_2 << arma::endl;

    //
    // test 3
    int test_3_dim = 5;
    arma::vec x_3 = arma::ones(test_3_dim,1);

    bool success_3 = optim::cg(x_3,unconstr_test_fn_3,NULL);

    if (success_3) {
        std::cout << "cg: test_3 completed successfully." << std::endl;
    } else {
        std::cout << "cg: test_3 completed unsuccessfully." << std::endl;
    }

    arma::cout << "cg: solution to test_3:\n" << x_3 << arma::endl;

    //
    // test 4
    arma::vec x_4 = arma::ones(2,1);

    bool success_4 = optim::cg(x_4,unconstr_test_fn_4,NULL);

    if (success_4) {
        std::cout << "cg: test_4 completed successfully." << std::endl;
    } else {
        std::cout << "cg: test_4 completed unsuccessfully." << std::endl;
    }

    arma::cout << "cg: solution to test_4:\n" << x_4 << arma::endl;

    return 0;
}
