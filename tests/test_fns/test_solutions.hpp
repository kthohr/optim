/*################################################################################
  ##
  ##   Copyright (C) 2016-2022 Keith O'Hara
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

#ifndef OPTIMLIB_TEST_SOLUTIONS
#define OPTIMLIB_TEST_SOLUTIONS

namespace unconstr_test_sols
{

ColVec_t test_1()
{
    ColVec_t ret(2);

    ret(0) = 2.25;
    ret(1) = -4.75;

    return ret;
}

//

ColVec_t test_2()
{
    return BMO_MATOPS_ONE_COLVEC(2);
}

//

ColVec_t test_3(const int n)
{
    return BMO_MATOPS_ZERO_COLVEC(n);
}

//

ColVec_t test_4()
{
    ColVec_t ret(2);

    ret(0) = 3.0;
    ret(1) = 0.5;

    return ret;
}

//

ColVec_t test_5()
{
    ColVec_t ret(2);

    ret(0) = 1.0;
    ret(1) = 3.0;

    return ret;
}

//

ColVec_t test_6()
{
    return BMO_MATOPS_ZERO_COLVEC(2);
}

//

ColVec_t test_7()
{
    return BMO_MATOPS_ZERO_COLVEC(2);
}

//

ColVec_t test_8()
{
    return BMO_MATOPS_ONE_COLVEC(2);
}

//

ColVec_t test_9()
{
    ColVec_t ret(2);

    ret(0) = -10.0;
    ret(1) =   1.0;

    return ret;
}

}

#endif
