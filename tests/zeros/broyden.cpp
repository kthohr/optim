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
 * Broyden tests
 */

#include "optim.hpp"
#include "./../test_fns/test_fns.hpp"

int main()
{
    
    std::cout << "\n     ***** Begin Broyden tests. *****     \n" << std::endl;

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

    //

    std::cout << "\n     ***** End Broyden tests. *****     \n" << std::endl;

    return 0;
}
