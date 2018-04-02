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
 * Newton tests
 */

#include "optim.hpp"
#include "./../test_fns/test_fns.hpp"

int main()
{
    std::cout << "\n     ***** Begin Newton tests. *****     \n" << std::endl;

    //
    // test 3

    int test_3_dim = 5;
    arma::vec x_3 = arma::ones(test_3_dim,1);

    bool success_3 = optim::newton(x_3,unconstr_test_fn_3_whess,nullptr);

    if (success_3) {
        std::cout << "\nnewton: test_3 completed successfully." << std::endl;
    } else {
        std::cout << "\nnewton: test_3 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_3:\n" \
              << arma::norm(x_3 - unconstr_test_sols::test_3(test_3_dim)) << std::endl;

    //

    std::cout << "\n     ***** End Newton tests. *****     \n" << std::endl;

    return 0;
}
