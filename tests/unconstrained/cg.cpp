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
    opt_params.cg_method = 5;

    arma::vec x_1 = arma::ones(2,1);

    bool success_1 = optim::cg(x_1,unconstr_test_fn_1,nullptr,opt_params);

    if (success_1) {
        std::cout << "cg: test_1 completed successfully." << std::endl;
    } else {
        std::cout << "cg: test_1 completed unsuccessfully." << std::endl;
    }

    arma::cout << "cg: solution to test_1:\n" << x_1 << arma::endl;

    //
    // test 2

    arma::vec x_2 = arma::zeros(2,1);

    bool success_2 = optim::cg(x_2,unconstr_test_fn_2,nullptr);

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

    bool success_3 = optim::cg(x_3,unconstr_test_fn_3,nullptr);

    if (success_3) {
        std::cout << "cg: test_3 completed successfully." << std::endl;
    } else {
        std::cout << "cg: test_3 completed unsuccessfully." << std::endl;
    }

    arma::cout << "cg: solution to test_3:\n" << x_3 << arma::endl;

    //
    // test 4
    arma::vec x_4 = arma::ones(2,1);

    bool success_4 = optim::cg(x_4,unconstr_test_fn_4,nullptr);

    if (success_4) {
        std::cout << "cg: test_4 completed successfully." << std::endl;
    } else {
        std::cout << "cg: test_4 completed unsuccessfully." << std::endl;
    }

    arma::cout << "cg: solution to test_4:\n" << x_4 << arma::endl;

    //
    // test 5
    optim::optim_opt_settings opt_params_5;
    opt_params_5.iter_max = 10000;
    opt_params_5.cg_method = 5;

    arma::vec x_5 = arma::zeros(2,1) + 2;

    bool success_5 = optim::cg(x_5,unconstr_test_fn_5,nullptr,opt_params_5);

    if (success_5) {
        std::cout << "cg: test_5 completed successfully." << std::endl;
    } else {
        std::cout << "cg: test_5 completed unsuccessfully." << std::endl;
    }

    arma::cout << "cg: solution to test_5:\n" << x_5 << arma::endl;

    //
    // for coverage

    optim::optim_opt_settings opt_settings;

    x_1 = arma::zeros(2,1);
    opt_settings.cg_method = 1;
    optim::cg(x_1,unconstr_test_fn_2,nullptr,opt_settings);

    arma::cout << "cg: solution to test_2 using cg_method = 1\n" << x_1 << arma::endl;

    x_1 = arma::zeros(2,1);
    opt_settings.cg_method = 2;

    optim::cg(x_1,unconstr_test_fn_2,nullptr,opt_settings);

    arma::cout << "cg: solution to test_2 using cg_method = 2\n" << x_1 << arma::endl;

    x_1 = arma::zeros(2,1);
    opt_settings.cg_method = 3;

    optim::cg(x_1,unconstr_test_fn_2,nullptr,opt_settings);

    arma::cout << "cg: solution to test_2 using cg_method = 3\n" << x_1 << arma::endl;

    x_1 = arma::zeros(2,1);
    opt_settings.cg_method = 4;
    optim::cg(x_1,unconstr_test_fn_2,nullptr,opt_settings);

    arma::cout << "cg: solution to test_2 using cg_method = 4\n" << x_1 << arma::endl;

    x_1 = arma::zeros(2,1);
    opt_settings.cg_method = 5;
    optim::cg(x_1,unconstr_test_fn_2,nullptr,opt_settings);

    arma::cout << "cg: solution to test_2 using cg_method = 5\n" << x_1 << arma::endl;
    
    x_1 = arma::zeros(2,1);
    opt_settings.cg_method = 6;
    optim::cg(x_1,unconstr_test_fn_2,nullptr,opt_settings);

    arma::cout << "cg: solution to test_2 using cg_method = 6\n" << x_1 << arma::endl;

    double val_out;

    optim::cg(x_1,unconstr_test_fn_1,nullptr);
    optim::cg(x_1,unconstr_test_fn_1,nullptr,val_out);
    optim::cg(x_1,unconstr_test_fn_1,nullptr,val_out,opt_settings);

    return 0;
}
