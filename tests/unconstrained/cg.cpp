/*################################################################################
  ##
  ##   Copyright (C) 2016-2018 Keith O'Hara
  ##
  ##   This file is part of the OptimLib C++ library.
  ##
  ##   OptimLib is free software: you can redistribute it and/or modify
  ##   it under the terms of the GNU General Public License as published by
  ##   the Free Software Foundation, either version 2 of the License, or
  ##   (at your option) any later version.
  ##
  ##   OptimLib is distributed in the hope that it will be useful,
  ##   but WITHOUT ANY WARRANTY; without even the implied warranty of
  ##   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  ##   GNU General Public License for more details.
  ##
  ################################################################################*/

/*
 * Nonlinear CG tests
 */

#include "optim.hpp"
#include "./../test_fns/test_fns.hpp"

int main()
{
    //
    // test 1
    optim::algo_settings settings_1;

    settings_1.iter_max = 2000;
    settings_1.conv_failure_switch = 1;
    settings_1.cg_method = 5;

    arma::vec x_1 = arma::ones(2,1);

    bool success_1 = optim::cg(x_1,unconstr_test_fn_1,nullptr,settings_1);

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
    optim::algo_settings settings_5;
    settings_5.iter_max = 10000;
    settings_5.cg_method = 5;

    arma::vec x_5 = arma::zeros(2,1) + 2;

    bool success_5 = optim::cg(x_5,unconstr_test_fn_5,nullptr,settings_5);

    if (success_5) {
        std::cout << "cg: test_5 completed successfully." << std::endl;
    } else {
        std::cout << "cg: test_5 completed unsuccessfully." << std::endl;
    }

    arma::cout << "cg: solution to test_5:\n" << x_5 << arma::endl;

    //
    // for coverage

    optim::algo_settings settings;

    x_1 = arma::zeros(2,1);
    settings.cg_method = 1;
    optim::cg(x_1,unconstr_test_fn_2,nullptr,settings);

    arma::cout << "cg: solution to test_2 using cg_method = 1\n" << x_1 << arma::endl;

    x_1 = arma::zeros(2,1);
    settings.cg_method = 2;

    optim::cg(x_1,unconstr_test_fn_2,nullptr,settings);

    arma::cout << "cg: solution to test_2 using cg_method = 2\n" << x_1 << arma::endl;

    x_1 = arma::zeros(2,1);
    settings.cg_method = 3;

    optim::cg(x_1,unconstr_test_fn_2,nullptr,settings);

    arma::cout << "cg: solution to test_2 using cg_method = 3\n" << x_1 << arma::endl;

    x_1 = arma::zeros(2,1);
    settings.cg_method = 4;
    optim::cg(x_1,unconstr_test_fn_2,nullptr,settings);

    arma::cout << "cg: solution to test_2 using cg_method = 4\n" << x_1 << arma::endl;

    x_1 = arma::zeros(2,1);
    settings.cg_method = 5;
    optim::cg(x_1,unconstr_test_fn_2,nullptr,settings);

    arma::cout << "cg: solution to test_2 using cg_method = 5\n" << x_1 << arma::endl;
    
    x_1 = arma::zeros(2,1);
    settings.cg_method = 6;
    optim::cg(x_1,unconstr_test_fn_2,nullptr,settings);

    arma::cout << "cg: solution to test_2 using cg_method = 6\n" << x_1 << arma::endl;

    //

    optim::algo_settings settings_2;

    settings_2.vals_bound = true;
    settings_2.lower_bounds = arma::zeros(2,1) - 4.5;
    settings_2.upper_bounds = arma::zeros(2,1) + 4.5;

    x_4 = arma::ones(2,1);
    
    success_4 = optim::cg(x_4,unconstr_test_fn_4,nullptr,settings_2);

    arma::cout << "cg: solution to test_4 with box constraints:\n" << x_4 << arma::endl;

    return 0;
}
