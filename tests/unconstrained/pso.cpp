/*################################################################################
  ##
  ##   Copyright (C) 2016-2023 Keith O'Hara
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
 * PSO tests
 */

#include "optim.hpp"
#include "./../test_fns/test_fns.hpp"

int main()
{
    
    std::cout << "\n     ***** Begin PSO tests. *****     \n" << std::endl;

    //
    // test 1

    optim::algo_settings_t settings_1;
    // settings_1.print_level = 4;

    ColVec_t x_1 = BMO_MATOPS_ONE_COLVEC(2);

    bool success_1 = optim::pso(x_1,unconstr_test_fn_1,nullptr,settings_1);

    if (success_1) {
        std::cout << "pso: test_1 completed successfully." << std::endl;
    } else {
        std::cout << "pso: test_1 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_1:\n" \
              << BMO_MATOPS_L2NORM(x_1 - unconstr_test_sols::test_1()) << std::endl;

    //
    // test 2

    ColVec_t x_2 = BMO_MATOPS_ZERO_COLVEC(2);

    bool success_2 = optim::pso(x_2,unconstr_test_fn_2,nullptr);

    if (success_2) {
        std::cout << "\npso: test_2 completed successfully." << std::endl;
    } else {
        std::cout << "\npso: test_2 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_2:\n" \
              << BMO_MATOPS_L2NORM(x_2 - unconstr_test_sols::test_2()) << std::endl;

    //
    // test 3

    int test_3_dim = 5;
    ColVec_t x_3 = BMO_MATOPS_ONE_COLVEC(test_3_dim);

    bool success_3 = optim::pso(x_3,unconstr_test_fn_3,nullptr);

    if (success_3) {
        std::cout << "\npso: test_3 completed successfully." << std::endl;
    } else {
        std::cout << "\npso: test_3 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_3:\n" \
              << BMO_MATOPS_L2NORM(x_3 - unconstr_test_sols::test_3(test_3_dim)) << std::endl;

    //
    // test 4

    ColVec_t x_4 = BMO_MATOPS_ONE_COLVEC(2);

    bool success_4 = optim::pso(x_4,unconstr_test_fn_4,nullptr);

    if (success_4) {
        std::cout << "\npso: test_4 completed successfully." << std::endl;
    } else {
        std::cout << "\npso: test_4 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_4:\n" \
              << BMO_MATOPS_L2NORM(x_4 - unconstr_test_sols::test_4()) << std::endl;

    //
    // test 6

    optim::algo_settings_t settings_6;

    settings_6.pso_settings.n_pop = 1000;

    unconstr_test_fn_6_data test_6_data;
    test_6_data.A = 10;

    ColVec_t x_6 = BMO_MATOPS_ARRAY_ADD_SCALAR(BMO_MATOPS_ZERO_COLVEC(2), 2);

    settings_6.pso_settings.initial_lb = BMO_MATOPS_ARRAY_ADD_SCALAR(x_6, -2.0);
    settings_6.pso_settings.initial_ub = BMO_MATOPS_ARRAY_ADD_SCALAR(x_6,  2.0);

    bool success_6 = optim::pso(x_6,unconstr_test_fn_6, &test_6_data, settings_6);

    if (success_6) {
        std::cout << "\npso: test_6 completed successfully." << std::endl;
    } else {
        std::cout << "\npso: test_6 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_6:\n" \
              << BMO_MATOPS_L2NORM(x_6 - unconstr_test_sols::test_6()) << std::endl;

    //
    // test 7

    ColVec_t x_7 = BMO_MATOPS_ONE_COLVEC(2);

    bool success_7 = optim::pso(x_7,unconstr_test_fn_7,nullptr);

    if (success_7) {
        std::cout << "\npso: test_7 completed successfully." << std::endl;
    } else {
        std::cout << "\npso: test_7 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_7:\n" \
              << BMO_MATOPS_L2NORM(x_7 - unconstr_test_sols::test_7()) << std::endl;

    //
    // test 8

    ColVec_t x_8 = BMO_MATOPS_ZERO_COLVEC(2);

    bool success_8 = optim::pso(x_8,unconstr_test_fn_8,nullptr);

    if (success_8) {
        std::cout << "\npso: test_8 completed successfully." << std::endl;
    } else {
        std::cout << "\npso: test_8 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_8:\n" \
              << BMO_MATOPS_L2NORM(x_8 - unconstr_test_sols::test_8()) << std::endl;

    //
    // test 9

    optim::algo_settings_t settings_9;
    
    settings_9.pso_settings.initial_lb = BMO_MATOPS_ZERO_COLVEC(2);
    settings_9.pso_settings.initial_lb(0) = -13;
    settings_9.pso_settings.initial_lb(1) = -2;

    settings_9.pso_settings.initial_ub = BMO_MATOPS_ZERO_COLVEC(2);
    settings_9.pso_settings.initial_ub(0) = -9;
    settings_9.pso_settings.initial_ub(1) = 2;

    settings_9.vals_bound = true;

    settings_9.lower_bounds = BMO_MATOPS_ZERO_COLVEC(2);
    settings_9.lower_bounds(0) = -15.0;
    settings_9.lower_bounds(1) = -3.0;

    settings_9.upper_bounds = BMO_MATOPS_ZERO_COLVEC(2);
    settings_9.upper_bounds(0) = 15.0;
    settings_9.upper_bounds(1) = 3.0;

    ColVec_t x_9 = BMO_MATOPS_ZERO_COLVEC(2);
    x_9(0) = -11.0;

    settings_9.pso_settings.n_gen = 4000;

    bool success_9 = optim::pso(x_9,unconstr_test_fn_9,nullptr,settings_9);

    if (success_9) {
        std::cout << "\npso: test_9 completed successfully." << std::endl;
    } else {
        std::cout << "\npso: test_9 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_9:\n" \
              << BMO_MATOPS_L2NORM(x_9 - unconstr_test_sols::test_9()) << std::endl;

    //
    // test 10

    optim::algo_settings_t settings_10;

    settings_10.pso_settings.center_particle = false;

    settings_10.vals_bound = true;
    settings_10.lower_bounds = BMO_MATOPS_ARRAY_ADD_SCALAR(BMO_MATOPS_ZERO_COLVEC(2), -10.0);
    settings_10.upper_bounds = BMO_MATOPS_ARRAY_ADD_SCALAR(BMO_MATOPS_ZERO_COLVEC(2),  10.0);

    ColVec_t x_10 = BMO_MATOPS_ZERO_COLVEC(2);

    settings_10.pso_settings.initial_lb = BMO_MATOPS_ARRAY_ADD_SCALAR(BMO_MATOPS_ZERO_COLVEC(2), -9.9);
    settings_10.pso_settings.initial_ub = BMO_MATOPS_ARRAY_ADD_SCALAR(BMO_MATOPS_ZERO_COLVEC(2),  9.9);

    settings_10.pso_settings.n_pop = 5000;
    settings_10.pso_settings.n_gen = 4000;

    bool success_10 = optim::pso(x_10,unconstr_test_fn_10,nullptr,settings_10);

    if (success_10) {
        std::cout << "\npso: test_10 completed successfully." << std::endl;
    } else {
        std::cout << "\npso: test_10 completed unsuccessfully." << std::endl;
    }

    BMO_MATOPS_COUT << "pso: solution to test_10:\n" << x_10 << "\n";

    //
    // for coverage

    optim::algo_settings_t settings;

    optim::pso(x_1,unconstr_test_fn_1,nullptr);
    optim::pso(x_1,unconstr_test_fn_1,nullptr,settings);

    x_7 = BMO_MATOPS_ARRAY_ADD_SCALAR(BMO_MATOPS_ZERO_COLVEC(2), 2.0);
    optim::pso(x_7,unconstr_test_fn_7,nullptr,settings);

    if (success_7) {
        std::cout << "\npso: test_7 completed successfully." << std::endl;
    } else {
        std::cout << "\npso: test_7 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_7:\n" \
              << BMO_MATOPS_L2NORM(x_7 - unconstr_test_sols::test_7()) << std::endl;

    //

    optim::algo_settings_t settings_2;

    settings_2.vals_bound = true;
    settings_2.lower_bounds = BMO_MATOPS_ARRAY_ADD_SCALAR(BMO_MATOPS_ZERO_COLVEC(2), -4.5);
    settings_2.upper_bounds = BMO_MATOPS_ARRAY_ADD_SCALAR(BMO_MATOPS_ZERO_COLVEC(2),  4.5);

    x_4 = BMO_MATOPS_ONE_COLVEC(2);
    
    success_4 = optim::pso(x_4,unconstr_test_fn_4,nullptr,settings_2);

    if (success_4) {
        std::cout << "\npso with box constraints: test_4 completed successfully." << std::endl;
    } else {
        std::cout << "\npso with box constraints: test_4 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_4:\n" \
              << BMO_MATOPS_L2NORM(x_4 - unconstr_test_sols::test_4()) << std::endl;

    //

    std::cout << "\n     ***** End PSO tests. *****     \n" << std::endl;

    return 0;
}
