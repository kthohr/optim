/*################################################################################
  ##
  ##   Copyright (C) 2016-2018 Keith O'Hara
  ##
  ##   This file is part of the OptimLib C++ library.
  ##
  ##   Licensed under the Apache License, Version 2.0 (the "License");
  ##   you may not use this file except in compliance with the License.
  ##   You may obtain a copy of the License at
  ##
  ##       http://www.apache.org/licenses/LICENSE-2.0
  ##
  ##   Unless required by applicable law or agreed to in writing, software
  ##   distributed under the License is distributed on an "AS IS" BASIS,
  ##   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ##   See the License for the specific language governing permissions and
  ##   limitations under the License.
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

    optim::algo_settings_t settings;

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
