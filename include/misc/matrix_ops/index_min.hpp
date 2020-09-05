/*################################################################################
  ##
  ##   Copyright (C) 2016-2020 Keith O'Hara
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

inline
size_t
index_min(const Vec_t& x)
{
    size_t x_n = OPTIM_MATOPS_SIZE(x);
    
    size_t min_ind = 0;
    double min_val = x(0);

    if (x_n > 1) {
        for (size_t i = 1; i < x_n; ++i) {
            if (x(i) < min_val) {
                min_val = x(i);
                min_ind = i;
            }
        }
    }

    return min_ind;
}
