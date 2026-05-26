"""
MPI regression test for SemiLagrangian tensor history traceback.

Tensor stress history is traced back through ``uw.function.global_evaluate``
before viscoelastic Stokes solves. Some rank partitions can trigger an MPI
collective mismatch in the FE evaluator path (``evalf=False``), so tensor
history defaults to the numerical evaluator path until the FE-path collective
bug is fixed at its source.
"""

import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.systems import ddt as ddt_module


pytestmark = [
    pytest.mark.level_2,
    pytest.mark.tier_a,
    pytest.mark.mpi(min_size=4),
    pytest.mark.timeout(60),
]


def test_tensor_history_traceback_completes_under_mpi():
    mesh = uw.meshing.StructuredQuadBox(elementRes=(16, 8))
    velocity = uw.discretisation.MeshVariable("V_trace", mesh, mesh.dim, degree=1)

    velocity.data[:, 0] = 0.05
    velocity.data[:, 1] = 0.0

    history = ddt_module.SemiLagrangian(
        mesh,
        psi_fn=sympy.zeros(mesh.dim, mesh.dim),
        V_fn=velocity.sym,
        vtype=uw.VarType.SYM_TENSOR,
        degree=1,
        continuous=True,
        order=1,
    )

    assert history.traceback_evalf is True

    x = history.psi_star[0].coords[:, 0]
    y = history.psi_star[0].coords[:, 1]
    initial_stress = np.zeros_like(history.psi_star[0].array)
    initial_stress[:, 0, 0] = 1.0 + x
    initial_stress[:, 0, 1] = 0.25 + 0.5 * y
    initial_stress[:, 1, 0] = initial_stress[:, 0, 1]
    initial_stress[:, 1, 1] = 2.0 - y
    history.set_initial_history([initial_stress], dt=0.01)

    history.update_pre_solve(0.01, store_result=False)

    local_finite = bool(np.isfinite(history.psi_star[0].array).all())
    assert all(uw.mpi.comm.allgather(local_finite))
