import numpy as np
import pytest

from defs import nb_dstn, params
from Paper_figures import get_fn_and_jacs
from Tools.SimpleNKAgent import SimpleNKAgent


@pytest.fixture(scope="module")
def agent():
    """Create a solved agent instance once per test session."""
    ag = SimpleNKAgent(**params)
    ag.update_solution_terminal()
    ag.solve(verbose=False)
    ag.build_transitions(newborn_dstn=nb_dstn)
    return ag


def test_precomputed_jac_values(agent):
    """
    Compare jacobian slices at a=0 and a=15 against precomputed reference.
    """
    jacs, _ = get_fn_and_jacs(agent, ["R"], ["c"], verbose=False)
    mat = jacs["R"]["c"]
    # At t=0 all entries should be zero
    assert mat[0, :5, :5] == pytest.approx(np.zeros((5, 5)))
    # Reference slice at t=15
    expected_15 = np.array(
        [
            [0.00041483, -0.00740778, -0.00698733, -0.00658487, -0.00619778],
            [0.00031589, 0.00068871, -0.00714858, -0.00674242, -0.00635390],
            [0.00023111, 0.00058503, 0.00094310, -0.00690763, -0.00651470],
            [0.00016180, 0.00049453, 0.00083277, 0.00117738, -0.00668573],
            [0.00010717, 0.00036586, 0.00068873, 0.00101716, 0.00135208],
        ]
    )
    assert mat[15, :5, :5] == pytest.approx(expected_15, abs=1e-8)
