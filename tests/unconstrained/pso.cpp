//
// PSO tests
//
// g++-mp-7 -O2 -Wall -std=c++11 -I/opt/local/include pso.cpp -o pso.test -L/opt/local/lib -loptim -framework Accelerate
// g++-mp-7 -O2 -Wall -std=c++11 -I./../../include pso.cpp -o pso.test -L./../.. -loptim -framework Accelerate
//

#include "optim.hpp"
#include "./../test_fns/test_fns.hpp"

int main()
{
    //
    // test 1
    arma::vec x_1 = arma::ones(2,1);

    bool success_1 = optim::pso(x_1,unconstr_test_fn_1,nullptr);

    if (success_1) {
        std::cout << "pso: test_1 completed successfully." << std::endl;
    } else {
        std::cout << "pso: test_1 completed unsuccessfully." << std::endl;
    }

    arma::cout << "pso: solution to test_1:\n" << x_1 << arma::endl;

    //
    // test 2

    arma::vec x_2 = arma::zeros(2,1);

    bool success_2 = optim::pso(x_2,unconstr_test_fn_2,nullptr);

    if (success_2) {
        std::cout << "pso: test_2 completed successfully." << std::endl;
    } else {
        std::cout << "pso: test_2 completed unsuccessfully." << std::endl;
    }

    arma::cout << "pso: solution to test_2:\n" << x_2 << arma::endl;

    //
    // test 3
    int test_3_dim = 5;
    arma::vec x_3 = arma::ones(test_3_dim,1);

    bool success_3 = optim::pso(x_3,unconstr_test_fn_3,nullptr);

    if (success_3) {
        std::cout << "pso: test_3 completed successfully." << std::endl;
    } else {
        std::cout << "pso: test_3 completed unsuccessfully." << std::endl;
    }

    arma::cout << "pso: solution to test_3:\n" << x_3 << arma::endl;

    //
    // test 4
    arma::vec x_4 = arma::ones(2,1);

    bool success_4 = optim::pso(x_4,unconstr_test_fn_4,nullptr);

    if (success_4) {
        std::cout << "pso: test_4 completed successfully." << std::endl;
    } else {
        std::cout << "pso: test_4 completed unsuccessfully." << std::endl;
    }

    arma::cout << "pso: solution to test_4:\n" << x_4 << arma::endl;

    //
    // test 6
    optim::opt_settings settings_6;

    settings_6.pso_lb = arma::zeros(2,1) - 2.0;
    settings_6.pso_ub = arma::zeros(2,1) + 2.0;
    settings_6.pso_n_pop = 1000;

    unconstr_test_fn_6_data test_6_data;
    test_6_data.A = 10;

    arma::vec x_6 = arma::ones(2,1) + 1.0;

    bool success_6 = optim::pso(x_6,unconstr_test_fn_6,&test_6_data,settings_6);

    if (success_6) {
        std::cout << "pso: test_6 completed successfully." << std::endl;
    } else {
        std::cout << "pso: test_6 completed unsuccessfully." << std::endl;
    }

    arma::cout << "pso: solution to test_6:\n" << x_6 << arma::endl;

    //
    // test 7
    arma::vec x_7 = arma::ones(2,1);

    bool success_7 = optim::pso(x_7,unconstr_test_fn_7,nullptr);

    if (success_7) {
        std::cout << "pso: test_7 completed successfully." << std::endl;
    } else {
        std::cout << "pso: test_7 completed unsuccessfully." << std::endl;
    }

    arma::cout << "pso: solution to test_7:\n" << x_7 << arma::endl;

    //
    // test 8
    arma::vec x_8 = arma::zeros(2,1);

    bool success_8 = optim::pso(x_8,unconstr_test_fn_8,nullptr);

    if (success_8) {
        std::cout << "pso: test_8 completed successfully." << std::endl;
    } else {
        std::cout << "pso: test_8 completed unsuccessfully." << std::endl;
    }

    arma::cout << "pso: solution to test_8:\n" << x_8 << arma::endl;

    //
    // test 9
    optim::opt_settings settings_9;
    
    settings_9.pso_lb = arma::zeros(2,1) - 2.0;
    settings_9.pso_ub = arma::zeros(2,1) + 2.0;

    arma::vec x_9 = arma::zeros(2,1);
    x_9(0) = -11.0;

    settings_9.pso_n_gen = 4000;

    bool success_9 = optim::pso(x_9,unconstr_test_fn_9,nullptr,settings_9);

    if (success_9) {
        std::cout << "pso: test_9 completed successfully." << std::endl;
    } else {
        std::cout << "pso: test_9 completed unsuccessfully." << std::endl;
    }

    arma::cout << "pso: solution to test_9:\n" << x_9 << arma::endl;

    //
    // test 10
    optim::opt_settings settings_10;

    settings_10.pso_center_particle = false;
    settings_10.pso_par_bounds = true;

    arma::vec x_10 = arma::zeros(2,1);

    settings_10.pso_lb = arma::zeros(2,1) - 10.0;
    settings_10.pso_ub = arma::zeros(2,1) + 10.0;

    settings_10.pso_n_pop = 5000;
    settings_10.pso_n_gen = 4000;

    bool success_10 = optim::pso(x_10,unconstr_test_fn_10,nullptr,settings_10);

    if (success_10) {
        std::cout << "pso: test_10 completed successfully." << std::endl;
    } else {
        std::cout << "pso: test_10 completed unsuccessfully." << std::endl;
    }

    arma::cout << "pso: solution to test_10:\n" << x_10 << arma::endl;

    //
    // for coverage

    optim::opt_settings settings;
    double val_out;

    optim::pso(x_1,unconstr_test_fn_1,nullptr);
    optim::pso(x_1,unconstr_test_fn_1,nullptr,settings);
    optim::pso(x_1,unconstr_test_fn_1,nullptr,val_out);
    optim::pso(x_1,unconstr_test_fn_1,nullptr,val_out,settings);

    x_7 = arma::ones(2,1) + 1.0;
    optim::pso(x_7,unconstr_test_fn_7,nullptr,settings);

    arma::cout << "pso: solution to test_7:\n" << x_7 << arma::endl;

    return 0;
}
