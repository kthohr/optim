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
 * BFGS tests
 */

#include "optim.hpp"
#include "./../test_fns/test_fns.hpp"

int main()
{

    std::cout << "\n     ***** Begin BFGS tests. *****     \n" << std::endl;

    //
    // test 1

    arma::vec x_1 = arma::ones(2,1);

    bool success_1 = optim::bfgs(x_1,unconstr_test_fn_1,nullptr);

    if (success_1) {
        std::cout << "bfgs: test_1 completed successfully." << std::endl;
    } else {
        std::cout << "bfgs: test_1 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_1:\n" \
              << arma::norm(x_1 - unconstr_test_sols::test_1()) << std::endl;

    //
    // test 2

    arma::vec x_2 = arma::zeros(2,1);

    bool success_2 = optim::bfgs(x_2,unconstr_test_fn_2,nullptr);

    if (success_2) {
        std::cout << "\nbfgs: test_2 completed successfully." << std::endl;
    } else {
        std::cout << "\nbfgs: test_2 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_2:\n" \
              << arma::norm(x_2 - unconstr_test_sols::test_2()) << std::endl;

    //
    // test 3
    
    int test_3_dim = 5;
    arma::vec x_3 = arma::ones(test_3_dim,1);

    bool success_3 = optim::bfgs(x_3,unconstr_test_fn_3,nullptr);

    if (success_3) {
        std::cout << "\nbfgs: test_3 completed successfully." << std::endl;
    } else {
        std::cout << "\nbfgs: test_3 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_3:\n" \
              << arma::norm(x_3 - unconstr_test_sols::test_3(test_3_dim)) << std::endl;

    //
    // test 4

    arma::vec x_4 = arma::ones(2,1);

    bool success_4 = optim::bfgs(x_4,unconstr_test_fn_4,nullptr);

    if (success_4) {
        std::cout << "\nbfgs: test_4 completed successfully." << std::endl;
    } else {
        std::cout << "\nbfgs: test_4 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_4:\n" \
              << arma::norm(x_4 - unconstr_test_sols::test_4()) << std::endl;

    //
    // test 5

    arma::vec x_5 = arma::zeros(2,1);

    bool success_5 = optim::bfgs(x_5,unconstr_test_fn_5,nullptr);

    if (success_5) {
        std::cout << "\nbfgs: test_5 completed successfully." << std::endl;
    } else {
        std::cout << "\nbfgs: test_5 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_5:\n" \
              << arma::norm(x_5 - unconstr_test_sols::test_5()) << std::endl;

    //
    // for coverage

    optim::algo_settings settings;

    optim::bfgs(x_1,unconstr_test_fn_1,nullptr);
    optim::bfgs(x_1,unconstr_test_fn_1,nullptr,settings);

    settings.vals_bound = true;
    settings.lower_bounds = arma::zeros(2,1) - 4.5;
    settings.upper_bounds = arma::zeros(2,1) + 4.5;

    x_4 = arma::ones(2,1);
    
    success_4 = optim::bfgs(x_4,unconstr_test_fn_4,nullptr,settings);

    if (success_4) {
        std::cout << "\nbfgs with box constraints: test_4 completed successfully." << std::endl;
    } else {
        std::cout << "\nbfgs with box constraints: test_4 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_4:\n" \
              << arma::norm(x_4 - unconstr_test_sols::test_4()) << std::endl;

    //

    std::cout << "\n     ***** End BFGS tests. *****     \n" << std::endl;

    return 0;
}
