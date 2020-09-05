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

/*
 * Optimization trace functions
 */

#ifndef OPTIM_TRACE_DEFS
#define OPTIM_TRACE_DEFS

#ifndef OPTIM_NO_TRACE

#define OPTIM_BFGS_TRACE(iter_in, grad_err_in, rel_sol_change_in, x_in, d_in, grad_in, s_in, y_in, W_in)    \
    if (print_level > 0) {                                                                                  \
        printf("\n");                                                                                       \
        std::cout << "  - Iteration:                   " << iter_in << "\n";                                \
        std::cout << "    Gradient error:              " << grad_err_in << "\n";                            \
        std::cout << "    Relative change in solution: " << rel_sol_change_in << "\n";                      \
                                                                                                            \
        if (print_level >= 2) {                                                                             \
            printf("\n");                                                                                   \
            std::cout << "    Current solution vector:\n";                                                  \
            OPTIM_MATOPS_COUT << OPTIM_MATOPS_TRANSPOSE(x_in) << "\n";                                      \
        }                                                                                                   \
                                                                                                            \
        if (print_level >= 3) {                                                                             \
            printf("\n");                                                                                   \
            std::cout << "    Direction vector:\n";                                                         \
            OPTIM_MATOPS_COUT << OPTIM_MATOPS_TRANSPOSE(d_in) << "\n";                                      \
            std::cout << "    Gradient vector:\n";                                                          \
            OPTIM_MATOPS_COUT << OPTIM_MATOPS_TRANSPOSE(grad_in) << "\n";                                   \
        }                                                                                                   \
                                                                                                            \
        if (print_level >= 4) {                                                                             \
            printf("\n");                                                                                   \
            std::cout << "    s:\n";                                                                        \
            OPTIM_MATOPS_COUT << s_in << "\n";                                                              \
            std::cout << "    y:\n";                                                                        \
            OPTIM_MATOPS_COUT << y_in << "\n";                                                              \
            std::cout << "    W:\n";                                                                        \
            OPTIM_MATOPS_COUT << W_in << "\n";                                                              \
        }                                                                                                   \
    }

#define OPTIM_CG_TRACE(iter_in, grad_err_in, rel_sol_change_in, x_in, d_in, grad_in, beta_in)               \
    if (print_level > 0) {                                                                                  \
        printf("\n");                                                                                       \
        std::cout << "  - Iteration:                   " << iter_in << "\n";                                \
        std::cout << "    Gradient error:              " << grad_err_in << "\n";                            \
        std::cout << "    Relative change in solution: " << rel_sol_change_in << "\n";                      \
                                                                                                            \
        if (print_level >= 2) {                                                                             \
            printf("\n");                                                                                   \
            std::cout << "    Current solution vector:\n";                                                  \
            OPTIM_MATOPS_COUT << OPTIM_MATOPS_TRANSPOSE(x_in) << "\n";                                      \
        }                                                                                                   \
                                                                                                            \
        if (print_level >= 3) {                                                                             \
            printf("\n");                                                                                   \
            std::cout << "    Direction vector:\n";                                                         \
            OPTIM_MATOPS_COUT << OPTIM_MATOPS_TRANSPOSE(d_in) << "\n";                                      \
            std::cout << "    Gradient vector:\n";                                                          \
            OPTIM_MATOPS_COUT << OPTIM_MATOPS_TRANSPOSE(grad_in) << "\n";                                   \
        }                                                                                                   \
                                                                                                            \
        if (print_level >= 4) {                                                                             \
            printf("\n");                                                                                   \
            std::cout << "    beta:\n";                                                                     \
            OPTIM_MATOPS_COUT << beta_in << "\n";                                                           \
        }                                                                                                   \
    }

#define OPTIM_DE_TRACE(iter_in, rel_objfn_change_in, min_objfn_val_running_in, min_objfn_val_check_in,      \
                       best_sol_running_in, X_next_in)                                                      \
    if (print_level > 0) {                                                                                  \
        printf("\n");                                                                                       \
        std::cout << "  - Iteration:        " << iter_in << "\n";                                           \
        std::cout << "    rel_objfn_change: " << rel_objfn_change_in << "\n";                               \
                                                                                                            \
        if (print_level >= 2) {                                                                             \
            printf("\n");                                                                                   \
            std::cout << "    Current lowest objective function value achieved: ";                          \
            std::cout << min_objfn_val_running_in << "\n";                                                  \
            std::cout << "    Stored lowest objective function value achieved:  ";                          \
            std::cout << min_objfn_val_check_in << "\n";                                                    \
            std::cout << "    Current best values:\n";                                                      \
            OPTIM_MATOPS_COUT << best_sol_running_in << "\n";                                               \
        }                                                                                                   \
                                                                                                            \
        if (print_level >= 3) {                                                                             \
            printf("\n");                                                                                   \
            std::cout << "    X:\n";                                                                        \
            OPTIM_MATOPS_COUT << X_next_in << "\n";                                                         \
        }                                                                                                   \
    }

#define OPTIM_DEPRMM_TRACE(iter_in, rel_objfn_change_in, min_objfn_val_running_in, min_objfn_val_check_in,  \
                           best_sol_running_in, X_next_in)                                                  \
    if (print_level > 0) {                                                                                  \
        printf("\n");                                                                                       \
        std::cout << "  - Iteration:        " << iter_in << "\n";                                           \
        std::cout << "    rel_objfn_change: " << rel_objfn_change_in << "\n";                               \
                                                                                                            \
        if (print_level >= 2) {                                                                             \
            printf("\n");                                                                                   \
            std::cout << "    Current lowest objective function value achieved: ";                          \
            std::cout << min_objfn_val_running_in << "\n";                                                  \
            std::cout << "    Stored lowest objective function value achieved:  ";                          \
            std::cout << min_objfn_val_check_in << "\n";                                                    \
            std::cout << "    Current best values:\n";                                                      \
            OPTIM_MATOPS_COUT << best_sol_running_in << "\n";                                               \
        }                                                                                                   \
                                                                                                            \
        if (print_level >= 3) {                                                                             \
            printf("\n");                                                                                   \
            std::cout << "    X:\n";                                                                        \
            OPTIM_MATOPS_COUT << X_next_in << "\n";                                                         \
        }                                                                                                   \
    }

#define OPTIM_GD_TRACE(iter_in, grad_err_in, rel_sol_change_in, x_in, d_in, grad_in,                        \
                       adam_vec_m_in, adam_vec_v_in)                                                        \
    if (print_level > 0) {                                                                                  \
        printf("\n");                                                                                       \
        std::cout << "  - Iteration:                   " << iter_in << "\n";                                \
        std::cout << "    Gradient error:              " << grad_err_in << "\n";                            \
        std::cout << "    Relative change in solution: " << rel_sol_change_in << "\n";                      \
                                                                                                            \
        if (print_level >= 2) {                                                                             \
            printf("\n");                                                                                   \
            std::cout << "    Current values:\n";                                                           \
            OPTIM_MATOPS_COUT << OPTIM_MATOPS_TRANSPOSE(x_in) << "\n";                                      \
        }                                                                                                   \
                                                                                                            \
        if (print_level >= 3) {                                                                             \
            printf("\n");                                                                                   \
            std::cout << "    Direction:\n";                                                                \
            OPTIM_MATOPS_COUT << OPTIM_MATOPS_TRANSPOSE(d_in) << "\n";                                      \
            std::cout << "    Gradient:\n";                                                                 \
            OPTIM_MATOPS_COUT << OPTIM_MATOPS_TRANSPOSE(grad_in) << "\n";                                   \
        }                                                                                                   \
                                                                                                            \
        if (print_level >= 4) {                                                                             \
            printf("\n");                                                                                   \
            if (settings.gd_settings.method == 3 || settings.gd_settings.method == 4) {                     \
                std::cout << "    adam_vec_v:\n";                                                           \
                OPTIM_MATOPS_COUT << OPTIM_MATOPS_TRANSPOSE(adam_vec_v_in) << "\n";                         \
            }                                                                                               \
                                                                                                            \
            if (settings.gd_settings.method == 5 || settings.gd_settings.method == 6 ||                     \
                    settings.gd_settings.method == 7) {                                                     \
                std::cout << "    adam_vec_m:\n";                                                           \
                OPTIM_MATOPS_COUT << OPTIM_MATOPS_TRANSPOSE(adam_vec_m_in) << "\n";                         \
                std::cout << "    adam_vec_v:\n";                                                           \
                OPTIM_MATOPS_COUT << OPTIM_MATOPS_TRANSPOSE(adam_vec_v_in) << "\n";                         \
            }                                                                                               \
        }                                                                                                   \
    }

#define OPTIM_LBFGS_TRACE(iter_in, grad_err_in, rel_sol_change_in, x_in, d_in, grad_in, s_mat_in, y_mat_in) \
    if (print_level > 0) {                                                                                  \
        printf("\n");                                                                                       \
        std::cout << "  - Iteration:                   " << iter_in << "\n";                                \
        std::cout << "    Gradient error:              " << grad_err_in << "\n";                            \
        std::cout << "    Relative change in solution: " << rel_sol_change_in << "\n";                      \
                                                                                                            \
        if (print_level >= 2) {                                                                             \
            printf("\n");                                                                                   \
            std::cout << "    Current values:\n";                                                           \
            OPTIM_MATOPS_COUT << OPTIM_MATOPS_TRANSPOSE(x_in) << "\n";                                      \
        }                                                                                                   \
                                                                                                            \
        if (print_level >= 3) {                                                                             \
            printf("\n");                                                                                   \
            std::cout << "    Direction:\n";                                                                \
            OPTIM_MATOPS_COUT << OPTIM_MATOPS_TRANSPOSE(d_in) << "\n";                                      \
            std::cout << "    Gradient:\n";                                                                 \
            OPTIM_MATOPS_COUT << OPTIM_MATOPS_TRANSPOSE(grad_in) << "\n";                                   \
        }                                                                                                   \
                                                                                                            \
        if (print_level >= 4) {                                                                             \
            printf("\n");                                                                                   \
            std::cout << "    s:\n";                                                                        \
            OPTIM_MATOPS_COUT << s_mat_in << "\n";                                                          \
            std::cout << "    y:\n";                                                                        \
            OPTIM_MATOPS_COUT << y_mat_in << "\n";                                                          \
        }                                                                                                   \
    }

#define OPTIM_NEWTON_TRACE(iter_in, grad_err_in, rel_sol_change_in, x_in, d_in, grad_in, H_in)              \
    if (print_level > 0) {                                                                                  \
        printf("\n");                                                                                       \
        std::cout << "  - Iteration:                   " << iter_in << "\n";                                \
        std::cout << "    Gradient error:              " << grad_err_in << "\n";                            \
        std::cout << "    Relative change in solution: " << rel_sol_change_in << "\n";                      \
                                                                                                            \
        if (print_level >= 2) {                                                                             \
            printf("\n");                                                                                   \
            std::cout << "    Current values:\n";                                                           \
            OPTIM_MATOPS_COUT << OPTIM_MATOPS_TRANSPOSE(x_in) << "\n";                                      \
        }                                                                                                   \
                                                                                                            \
        if (print_level >= 3) {                                                                             \
            printf("\n");                                                                                   \
            std::cout << "    Direction:\n";                                                                \
            OPTIM_MATOPS_COUT << OPTIM_MATOPS_TRANSPOSE(d_in) << "\n";                                      \
            std::cout << "    Gradient:\n";                                                                 \
            OPTIM_MATOPS_COUT << OPTIM_MATOPS_TRANSPOSE(grad_in) << "\n";                                   \
        }                                                                                                   \
                                                                                                            \
        if (print_level >= 4) {                                                                             \
            printf("\n");                                                                                   \
            OPTIM_MATOPS_COUT << "    Hessian:\n" << H_in << "\n";                                          \
        }                                                                                                   \
    }

#define OPTIM_NM_TRACE(iter_in, min_val_in, rel_objfn_change_in, rel_sol_change_in,                         \
                       simplex_fn_vals_in, simplex_points_in)                                               \
    if (print_level > 0) {                                                                                  \
        printf("\n");                                                                                       \
        std::cout << "  - Iteration:                          " << iter_in << "\n";                         \
        std::cout << "    Minimum simplex objective value:    " << min_val_in << "\n";                      \
        std::cout << "    Relative change in objective value: " << rel_objfn_change_in << "\n";             \
        std::cout << "    Relative change in solution:        " << rel_sol_change_in << "\n";               \
                                                                                                            \
        if (print_level >= 2) {                                                                             \
            printf("\n");                                                                                   \
            std::cout << "    Current optimal input values:\n";                                             \
            OPTIM_MATOPS_COUT << simplex_points_in.row(index_min(simplex_fn_vals_in)) << "\n";              \
        }                                                                                                   \
                                                                                                            \
        if (print_level >= 3) {                                                                             \
            printf("\n");                                                                                   \
            std::cout << "    Objective function value at each vertex:\n";                                  \
            OPTIM_MATOPS_COUT << OPTIM_MATOPS_TRANSPOSE(simplex_fn_vals_in) << "\n";                        \
            std::cout << "    Simplex matrix:\n";                                                           \
            OPTIM_MATOPS_COUT << OPTIM_MATOPS_TRANSPOSE(simplex_points_in) << "\n";                         \
        }                                                                                                   \
    }

#define OPTIM_PSO_TRACE(iter_in, rel_objfn_change_in, min_objfn_val_running_in, min_objfn_val_check_in,     \
                        best_sol_running_in, P_in)                                                          \
    if (print_level > 0) {                                                                                  \
        printf("\n");                                                                                       \
        std::cout << "  - Iteration:                          " << iter_in << "\n";                         \
        std::cout << "    Relative change in objective value: " << rel_objfn_change_in << "\n";             \
                                                                                                            \
        if (print_level >= 2) {                                                                             \
            printf("\n");                                                                                   \
            std::cout << "    Current lowest objective function value achieved: ";                          \
            std::cout << min_objfn_val_running_in << "\n";                                                  \
            std::cout << "    Stored lowest objective function value achieved:  ";                          \
            std::cout << min_objfn_val_check_in << "\n";                                                    \
            std::cout << "    Current best values:\n";                                                      \
            OPTIM_MATOPS_COUT << best_sol_running_in << "\n";                                               \
        }                                                                                                   \
                                                                                                            \
        if (print_level >= 3) {                                                                             \
            printf("\n");                                                                                   \
            std::cout << "    P:\n";                                                                        \
            OPTIM_MATOPS_COUT << P_in << "\n";                                                              \
        }                                                                                                   \
    }

#define OPTIM_PSODV_TRACE(iter_in, rel_objfn_change_in, min_objfn_val_running_in, min_objfn_val_check_in,   \
                        best_sol_running_in, par_w_in, P_in)                                                \
    if (print_level > 0) {                                                                                  \
        printf("\n");                                                                                       \
        std::cout << "  - Iteration:                          " << iter_in << "\n";                         \
        std::cout << "    Relative change in objective value: " << rel_objfn_change_in << "\n";             \
                                                                                                            \
        if (print_level >= 2) {                                                                             \
            printf("\n");                                                                                   \
            std::cout << "    Current lowest objective function value achieved: ";                          \
            std::cout << min_objfn_val_running_in << "\n";                                                  \
            std::cout << "    Stored lowest objective function value achieved:  ";                          \
            std::cout << min_objfn_val_check_in << "\n";                                                    \
            std::cout << "    Current best values:\n";                                                      \
            OPTIM_MATOPS_COUT << best_sol_running_in << "\n";                                               \
        }                                                                                                   \
                                                                                                            \
        if (print_level >= 3) {                                                                             \
            printf("\n");                                                                                   \
            std::cout << "    w: ";                                                                         \
            std::cout << par_w_in << "\n";                                                                  \
            std::cout << "    P:\n";                                                                        \
            OPTIM_MATOPS_COUT << P_in << "\n";                                                              \
        }                                                                                                   \
    }

#define OPTIM_BROYDEN_TRACE(iter_in, rel_objfn_change_in, rel_sol_change_in, x_in, d_in, objfn_vec_in,      \
                            s_in, y_in, B_in)                                                               \
    if (print_level > 0) {                                                                                  \
        printf("\n");                                                                                       \
        std::cout << "  - Iteration:                          " << iter_in << "\n";                         \
        std::cout << "    Relative change in objective value: " << rel_objfn_change_in << "\n";             \
        std::cout << "    Relative change in solution:        " << rel_sol_change_in << "\n";               \
                                                                                                            \
        if (print_level >= 2) {                                                                             \
            printf("\n");                                                                                   \
            std::cout << "    Current values:\n";                                                           \
            OPTIM_MATOPS_COUT << OPTIM_MATOPS_TRANSPOSE(x_in) << "\n";                                      \
        }                                                                                                   \
                                                                                                            \
        if (print_level >= 3) {                                                                             \
            printf("\n");                                                                                   \
            std::cout << "    Direction:\n";                                                                \
            OPTIM_MATOPS_COUT << OPTIM_MATOPS_TRANSPOSE(d_in) << "\n";                                      \
            std::cout << "    f(x):\n";                                                                     \
            OPTIM_MATOPS_COUT << OPTIM_MATOPS_TRANSPOSE(objfn_vec_in) << "\n";                              \
        }                                                                                                   \
                                                                                                            \
        if (print_level >= 4) {                                                                             \
            printf("\n");                                                                                   \
            std::cout << "    s:\n";                                                                        \
            OPTIM_MATOPS_COUT << s_in << "\n";                                                              \
            std::cout << "    y:\n";                                                                        \
            OPTIM_MATOPS_COUT << y_in << "\n";                                                              \
            std::cout << "    B:\n";                                                                        \
            OPTIM_MATOPS_COUT << B_in << "\n";                                                              \
        }                                                                                                   \
    }

#define OPTIM_BROYDEN_DF_TRACE(iter_in, rel_objfn_change_in, rel_sol_change_in, x_in, d_in, objfn_vec_in,   \
                               lambda_in, s_in, y_in, B_in)                                                 \
    if (print_level > 0) {                                                                                  \
        printf("\n");                                                                                       \
        std::cout << "  - Iteration:                          " << iter_in << "\n";                         \
        std::cout << "    Relative change in objective value: " << rel_objfn_change_in << "\n";             \
        std::cout << "    Relative change in solution:        " << rel_sol_change_in << "\n";               \
                                                                                                            \
        if (print_level >= 2) {                                                                             \
            printf("\n");                                                                                   \
            std::cout << "    Current values:\n";                                                           \
            OPTIM_MATOPS_COUT << OPTIM_MATOPS_TRANSPOSE(x_in) << "\n";                                      \
        }                                                                                                   \
                                                                                                            \
        if (print_level >= 3) {                                                                             \
            printf("\n");                                                                                   \
            std::cout << "    Direction:\n";                                                                \
            OPTIM_MATOPS_COUT << OPTIM_MATOPS_TRANSPOSE(d_in) << "\n";                                      \
            std::cout << "    f(x):\n";                                                                     \
            OPTIM_MATOPS_COUT << OPTIM_MATOPS_TRANSPOSE(objfn_vec_in) << "\n";                              \
        }                                                                                                   \
                                                                                                            \
        if (print_level >= 4) {                                                                             \
            printf("\n");                                                                                   \
            std::cout << "    lambda: " << lambda_in << "\n";                                               \
            std::cout << "    s:\n";                                                                        \
            OPTIM_MATOPS_COUT << s_in << "\n";                                                              \
            std::cout << "    y:\n";                                                                        \
            OPTIM_MATOPS_COUT << y_in << "\n";                                                              \
            std::cout << "    B:\n";                                                                        \
            OPTIM_MATOPS_COUT << B_in << "\n";                                                              \
        }                                                                                                   \
    }

#else

#define OPTIM_BFGS_TRACE(iter_in, grad_err_in, rel_sol_change_in, x_in, d_in, grad_in, s_in, y_in, W_in)

#define OPTIM_CG_TRACE(iter_in, grad_err_in, rel_sol_change_in, x_in, d_in, grad_in, beta_in)

#define OPTIM_DE_TRACE(iter_in, rel_objfn_change_in, min_objfn_val_running_in, min_objfn_val_check_in,      \
                       best_sol_running_in, X_next_in)

#define OPTIM_DEPRMM_TRACE(iter_in, rel_objfn_change_in, min_objfn_val_running_in, min_objfn_val_check_in,  \
                           best_sol_running_in, X_next_in)

#define OPTIM_GD_TRACE(iter_in, grad_err_in, rel_sol_change_in, x_in, d_in, grad_in,                        \
                       adam_vec_m_in, adam_vec_v_in)

#define OPTIM_LBFGS_TRACE(iter_in, grad_err_in, rel_sol_change_in, x_in, d_in, grad_in, s_mat_in, y_mat_in)

#define OPTIM_NEWTON_TRACE(iter_in, grad_err_in, rel_sol_change_in, x_in, d_in, grad_in, H_in)

#define OPTIM_NM_TRACE(iter_in, min_val_in, rel_objfn_change_in, simplex_fn_vals_in, simplex_points_in)

#define OPTIM_PSO_TRACE(iter_in, rel_objfn_change_in, min_objfn_val_running_in, min_objfn_val_check_in,     \
                        best_sol_running_in, X_next_in)

#define OPTIM_PSODV_TRACE(iter_in, rel_objfn_change_in, min_objfn_val_running_in, min_objfn_val_check_in,   \
                        best_sol_running_in, par_w_in, P_in)

#define OPTIM_BROYDEN_TRACE(iter_in, rel_objfn_change_in, rel_sol_change_in, x_in, d_in, objfn_vec_in,      \
                            s_in, y_in, B_in)

#define OPTIM_BROYDEN_DF_TRACE(iter_in, rel_objfn_change_in, rel_sol_change_in, x_in, d_in, objfn_vec_in,   \
                               lambda_in, s_in, y_in, B_in)

#endif

#endif

