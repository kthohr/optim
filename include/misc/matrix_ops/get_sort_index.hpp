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
VecInt_t
get_sort_index(const Vec_t& vec_in)
{
    const size_t n = OPTIM_MATOPS_SIZE(vec_in);

    std::vector<size_t> idx(n, static_cast<size_t>(0));
    std::iota(idx.begin(), idx.end(), static_cast<size_t>(0));

    sort(idx.begin(), idx.end(),
         [&vec_in](size_t i1, size_t i2) {return vec_in(i1) < vec_in(i2);});

    VecInt_t out_vec(n);

    for (size_t i = 0; i < n; ++i) {
        out_vec(i) = idx[i];
    }

    return out_vec;
}
