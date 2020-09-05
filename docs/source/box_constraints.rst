.. Copyright (c) 2016-2020 Keith O'Hara

   Distributed under the terms of the Apache License, Version 2.0.

   The full license is in the file LICENSE, distributed with this software.

Box Constraints
===============

This section provides implementation details for how OptimLib handles box constraints.

The problem is to transform

.. math::

    \min_{x \in X} f(x)

where :math:`X` is a subset of :math:`\mathbb{R}^d`, to

.. math::

    \min_{y \in \mathbb{R}^d} f(g^{-1}(y))

using a smooth, invertible mapping :math:`g: X \to \mathbb{R}^d`.

OptimLib uses allows the user to specify upper and lower bounds for each element of the input vector, :math:`x_j \in [a_j, b_j]`, and uses the following specification for :math:`g`:

.. math::

    g(x_j) = \ln \left( \frac{x_j - a_j}{b_j - x_j} \right)

with corresponding inverse:

.. math::

    g^{-1}(y_j) = \frac{a_j + b_j \exp(y_j)}{1 + \exp(y_j)}

The gradient vector is then:

.. math::

    \nabla_y f(g^{-1}(y)) = J(y) [\nabla_{x = g^{-1}(y)} f]

where :math:`J(y)` is a :math:`d \times d` diagonal matrix with typical element:

.. math::

    J_{j,j} = \frac{d}{d y_j} g^{-1}(y_j) = \frac{\exp( y_j ) (b_j - a_j)}{(1 + \exp(y_j))^2}
