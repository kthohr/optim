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

#ifndef OPTIMLIB_TEST_SOLUTIONS
#define OPTIMLIB_TEST_SOLUTIONS

namespace unconstr_test_sols
{

arma::vec test_1()
{
    arma::vec ret(2);

    ret(0) = 2.25;
    ret(1) = -4.75;

    return ret;
}

//

arma::vec test_2()
{
    return arma::ones(2,1);
}

//

arma::vec test_3(const int n)
{
    return arma::zeros(n,1);
}

//

arma::vec test_4()
{
    arma::vec ret(2);

    ret(0) = 3.0;
    ret(1) = 0.5;

    return ret;
}

//

arma::vec test_5()
{
    arma::vec ret(2);

    ret(0) = 1.0;
    ret(1) = 3.0;

    return ret;
}

//

arma::vec test_6()
{
    return arma::zeros(2,1);
}

//

arma::vec test_7()
{
    return arma::zeros(2,1);
}

//

arma::vec test_8()
{
    return arma::ones(2,1);
}

//

arma::vec test_9()
{
    arma::vec ret(2);

    ret(0) = -10.0;
    ret(1) =   1.0;

    return ret;
}

}

#endif
