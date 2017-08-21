/*################################################################################
  ##
  ##   Copyright (C) 2016-2017 Keith O'Hara
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

#ifndef OPTIMLIB_INCLUDES
#define OPTIMLIB_INCLUDES

#ifdef USE_RCPP_ARMADILLO
    #include <RcppArmadillo.h>
#else
    #ifndef ARMA_DONT_USE_WRAPPER
        #define ARMA_DONT_USE_WRAPPER
    #endif
    #include "armadillo"
#endif

#include "misc/optim_options.hpp"

namespace optim
{
    // structs
    #include "misc/optim_structs.hpp"

    // misc files
    #include "misc/misc.hpp"

    // line search
    #include "line_search/more_thuente.hpp"

    // unconstrained optimization
    #include "unconstrained/bfgs.hpp"
    #include "unconstrained/lbfgs.hpp" 
    #include "unconstrained/cg.hpp"
    #include "unconstrained/de.hpp"
    #include "unconstrained/de_prmm.hpp"
    #include "unconstrained/nm.hpp"
    #include "unconstrained/pso.hpp"
    #include "unconstrained/pso_dv.hpp"

    // constrained optimization
    #include "constrained/sumt.hpp"

    // solving systems of nonlinear equations
    #include "zeros/broyden.hpp"
}

#endif
