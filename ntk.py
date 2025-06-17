# Copyright 2021 Predicitve Intelligence Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax.numpy as np
from jax import jacobian, eval_shape, lax
from jax.tree_util import tree_map, tree_reduce
import operator


def canonicalize_axis(axes, ndim):
    """Turn possibly-negative axes into non-negative indices in [0,ndim)."""
    return tuple(a % ndim for a in axes)

def dot_general(x, y, contract_axes, diagonal_axes):
    """
    A thin wrapper around jax.lax.dot_general to sum over exactly
    the same axes on both x and y (no batch dims).
    """
    # we treat all `contract_axes` on both x and y as contracting dims,
    # with no batch dims:
    dimension_numbers = ((list(contract_axes), list(contract_axes)), ([], []))
    return lax.dot_general(x, y, dimension_numbers)

def sum_and_contract(j1, j2, output_ndim):
    """
    Given two PyTrees of Jacobians, each array shaped
      ( *output_dims, *param_dims ),
    sum ⟨J1, J2⟩ over both the output axes (trace_axes) and
    all the parameter axes.
    """
    diagonal_axes = canonicalize_axis((), output_ndim)
    trace_axes    = canonicalize_axis((-1,), output_ndim)
    def contract(x, y):
        param_axes    = tuple(range(output_ndim, x.ndim))
        contract_axes = trace_axes + param_axes
        return dot_general(x, y, contract_axes, diagonal_axes)
    return tree_reduce(operator.add, tree_map(contract, j1, j2))

def compute_ntk(f1, f2, x1, x2, params):
    """
    NTK(f1,f2)_{i,j} = <  f1(params, x1[i]),  f2(params, x2[j])>
    """
    j1  = jacobian(f1)(params, *x1)
    j2  = jacobian(f2)(params, *x2)
    fx1 = eval_shape(f1, params, *x1)
    return sum_and_contract(j1, j2, fx1.ndim)
