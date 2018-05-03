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

#ifndef OPTIMLIB_INCLUDES
#define OPTIMLIB_INCLUDES

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
    #include "unconstrained/newton.hpp"
    #include "unconstrained/cg.hpp"
    #include "unconstrained/gd.hpp"
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
