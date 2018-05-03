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
 * PSO-DV tests
 */

#include "optim.hpp"
#include "./../test_fns/test_fns.hpp"

int main()
{
    
    std::cout << "\n     ***** Begin PSO-DV tests. *****     \n" << std::endl;

    //
    // test 1

    arma::vec x_1 = arma::ones(2,1);

    bool success_1 = optim::pso_dv(x_1,unconstr_test_fn_1,nullptr);

    if (success_1) {
        std::cout << "pso_dv: test_1 completed successfully." << std::endl;
    } else {
        std::cout << "pso_dv: test_1 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_1:\n" \
              << arma::norm(x_1 - unconstr_test_sols::test_1()) << std::endl;

    //
    // test 2

    arma::vec x_2 = arma::zeros(2,1);

    bool success_2 = optim::pso_dv(x_2,unconstr_test_fn_2,nullptr);

    if (success_2) {
        std::cout << "\npso_dv: test_2 completed successfully." << std::endl;
    } else {
        std::cout << "\npso_dv: test_2 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_2:\n" \
              << arma::norm(x_2 - unconstr_test_sols::test_2()) << std::endl;

    //
    // test 3

    int test_3_dim = 5;
    arma::vec x_3 = arma::ones(test_3_dim,1);

    bool success_3 = optim::pso_dv(x_3,unconstr_test_fn_3,nullptr);

    if (success_3) {
        std::cout << "\npso_dv: test_3 completed successfully." << std::endl;
    } else {
        std::cout << "\npso_dv: test_3 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_3:\n" \
              << arma::norm(x_3 - unconstr_test_sols::test_3(test_3_dim)) << std::endl;

    //
    // test 4

    arma::vec x_4 = arma::ones(2,1);

    bool success_4 = optim::pso_dv(x_4,unconstr_test_fn_4,nullptr);

    if (success_4) {
        std::cout << "\npso_dv: test_4 completed successfully." << std::endl;
    } else {
        std::cout << "\npso_dv: test_4 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_4:\n" \
              << arma::norm(x_4 - unconstr_test_sols::test_4()) << std::endl;

    //
    // test 6

    optim::algo_settings_t settings_6;

    settings_6.pso_n_pop = 1000;

    unconstr_test_fn_6_data test_6_data;
    test_6_data.A = 10;

    arma::vec x_6 = arma::ones(2,1) + 1.0;
    
    settings_6.pso_initial_lb = x_6 - 2.0;
    settings_6.pso_initial_ub = x_6 + 2.0;

    bool success_6 = optim::pso_dv(x_6,unconstr_test_fn_6,&test_6_data,settings_6);

    if (success_6) {
        std::cout << "\npso_dv: test_6 completed successfully." << std::endl;
    } else {
        std::cout << "\npso_dv: test_6 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_6:\n" \
              << arma::norm(x_6 - unconstr_test_sols::test_6()) << std::endl;

    //
    // test 7

    arma::vec x_7 = arma::ones(2,1);

    bool success_7 = optim::pso_dv(x_7,unconstr_test_fn_7,nullptr);

    if (success_7) {
        std::cout << "\npso_dv: test_7 completed successfully." << std::endl;
    } else {
        std::cout << "\npso_dv: test_7 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_7:\n" \
              << arma::norm(x_7 - unconstr_test_sols::test_7()) << std::endl;

    //
    // test 8

    arma::vec x_8 = arma::zeros(2,1);

    bool success_8 = optim::pso_dv(x_8,unconstr_test_fn_8,nullptr);

    if (success_8) {
        std::cout << "\npso_dv: test_8 completed successfully." << std::endl;
    } else {
        std::cout << "\npso_dv: test_8 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_8:\n" \
              << arma::norm(x_8 - unconstr_test_sols::test_8()) << std::endl;

    //
    // test 9
    optim::algo_settings_t settings_9;
    
    settings_9.pso_initial_lb = arma::zeros(2,1);
    settings_9.pso_initial_lb(0) = -13;
    settings_9.pso_initial_lb(1) = -2;

    settings_9.pso_initial_ub = arma::zeros(2,1);
    settings_9.pso_initial_ub(0) = -9;
    settings_9.pso_initial_ub(1) = 2;

    settings_9.vals_bound = true;

    settings_9.lower_bounds = arma::zeros(2,1);
    settings_9.lower_bounds(0) = -15.0;
    settings_9.lower_bounds(1) = -3.0;

    settings_9.upper_bounds = arma::zeros(2,1);
    settings_9.upper_bounds(0) = 15.0;
    settings_9.upper_bounds(1) = 3.0;

    arma::vec x_9 = arma::zeros(2,1);
    x_9(0) = -11.0;

    settings_9.pso_n_gen = 4000;

    bool success_9 = optim::pso_dv(x_9,unconstr_test_fn_9,nullptr,settings_9);

    if (success_9) {
        std::cout << "\npso_dv: test_9 completed successfully." << std::endl;
    } else {
        std::cout << "\npso_dv: test_9 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_9:\n" \
              << arma::norm(x_9 - unconstr_test_sols::test_9()) << std::endl;

    //
    // test 10

    optim::algo_settings_t settings_10;

    settings_10.pso_center_particle = false;

    settings_10.vals_bound = true;
    settings_10.lower_bounds = arma::zeros(2,1) - 10.0;
    settings_10.upper_bounds = arma::zeros(2,1) + 10.0;

    arma::vec x_10 = arma::zeros(2,1);

    settings_10.pso_initial_lb = arma::zeros(2,1) - 9.9;
    settings_10.pso_initial_ub = arma::zeros(2,1) + 9.9;

    settings_10.pso_n_pop = 5000;
    settings_10.pso_n_gen = 4000;

    bool success_10 = optim::pso_dv(x_10,unconstr_test_fn_10,nullptr,settings_10);

    if (success_10) {
        std::cout << "\npso_dv: test_10 completed successfully." << std::endl;
    } else {
        std::cout << "\npso_dv: test_10 completed unsuccessfully." << std::endl;
    }

    arma::cout << "pso_dv: solution to test_10:\n" << x_10 << arma::endl;

    //
    // for coverage

    optim::algo_settings_t settings;

    optim::pso_dv(x_1,unconstr_test_fn_1,nullptr);
    optim::pso_dv(x_1,unconstr_test_fn_1,nullptr,settings);

    x_7 = arma::ones(2,1) + 1.0;
    optim::pso_dv(x_7,unconstr_test_fn_7,nullptr,settings);

    if (success_7) {
        std::cout << "\npso_dv: test_7 completed successfully." << std::endl;
    } else {
        std::cout << "\npso_dv: test_7 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_7:\n" \
              << arma::norm(x_7 - unconstr_test_sols::test_7()) << std::endl;

    //

    optim::algo_settings_t settings_2;

    settings_2.vals_bound = true;
    settings_2.lower_bounds = arma::zeros(2,1) - 4.5;
    settings_2.upper_bounds = arma::zeros(2,1) + 4.5;

    x_4 = arma::ones(2,1);
    
    success_4 = optim::pso_dv(x_4,unconstr_test_fn_4,nullptr,settings_2);

    if (success_4) {
        std::cout << "\npso_dv with box constraints: test_4 completed successfully." << std::endl;
    } else {
        std::cout << "\npso_dv with box constraints: test_4 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_4:\n" \
              << arma::norm(x_4 - unconstr_test_sols::test_4()) << std::endl;

    //

    std::cout << "\n     ***** End PSO-DV tests. *****     \n" << std::endl;

    return 0;
}