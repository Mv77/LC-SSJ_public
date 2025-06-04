from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from HARK.core import AgentType, Parameters, make_one_period_oo_solver
from HARK.metric import MetricObject
from HARK.utilities import make_grid_exp_mult
from numba import float64, guvectorize, jit
from sequence_jacobian.grids import markov_rouwenhorst

# Project-defined tools
from Tools.transition_matrices import DiscreteTransitions, TransitionMatrix


# %% Interpolators
@guvectorize(
    [(float64[:], float64[:], float64[:], float64[:])],
    "(n),(n),(p)->(p)",
    nopython=True,
)
def interp_monotone(m, c, m_new, c_new):
    n = m.shape[0]
    p = m_new.shape[0]

    # Precompute differences
    m_difs = np.empty(n - 1, dtype=float64)
    for i in range(n - 1):
        m_difs[i] = m[i + 1] - m[i]

    # Start interpolating
    ind = 0

    # Borrowing constraint at the lower bound
    while ind < p and m_new[ind] <= m[0]:
        c_new[ind] = m_new[ind]
        ind += 1

    # Actual interpolation
    upp_ind = 1
    while ind < p:
        # Increase upper index until a greater entry
        while upp_ind < (n - 1) and m[upp_ind] < m_new[ind]:
            upp_ind += 1
        # Interpolation weight
        w = (m_new[ind] - m[upp_ind - 1]) / m_difs[upp_ind - 1]
        c_new[ind] = (1.0 - w) * c[upp_ind - 1] + w * c[upp_ind]
        ind += 1


@guvectorize(
    [(float64[:], float64[:], float64[:, :])],
    "(a),(g)->(a,g)",
    nopython=True,
    target="parallel",
)
def mass_to_1d_grid_monotone(points_row, grid, mass_out):
    # Number of points in the row
    A = points_row.shape[0]
    # Number of grid cells
    G = grid.shape[0]

    # Zero out the output array (guvectorize reuses memory)
    for i in range(A):
        for j in range(G):
            mass_out[i, j] = 0.0

    # Precompute grid differences once
    difs = np.empty(G - 1, dtype=float64)
    for i in range(G - 1):
        difs[i] = grid[i + 1] - grid[i]

    # Single-pass index for monotonic points
    ind = 1
    for ai in range(A):
        a_t = points_row[ai]
        # Move 'ind' forward while grid[ind] < a_t
        while ind < (G - 1) and grid[ind] < a_t:
            ind += 1

        # Fraction of mass allocated to the upper bin
        num = a_t - grid[ind - 1]
        den = difs[ind - 1]
        frac = num / den

        mass_out[ai, ind - 1] += 1.0 - frac
        mass_out[ai, ind] += frac


# %% Auxiliary simple functions
@jit
def crra_inv(u, crra):
    return np.power(u * (1 - crra), 1 / (1 - crra))


@jit
def crra_marg(c, crra):
    return np.power(c, -crra)


@jit
def crra_marg_inv(c, crra):
    return np.power(c, -1.0 / crra)


@jit
def beq_func(a, beq_inten, beq_shift, crra):
    return beq_inten * np.power(a + beq_shift, 1 - crra) / (1 - crra)


@jit
def marg_beq_func(a, beq_inten, beq_shift, crra):
    return beq_inten * np.power(a + beq_shift, -crra)


# %% Transition matrix class for the model
class NKtmat(TransitionMatrix):
    def __init__(self, a_tmat, z_tmat):
        super().__init__()
        self.a_tmat = a_tmat
        self.z_tmat = z_tmat

    def postmult(self, out):
        # Apply productivity transition
        exp = np.einsum("...mk,jm", out, self.z_tmat)
        # Apply asset transition
        exp = np.einsum("...jk,jlk->...jl", exp, self.a_tmat)
        return exp

    def premult(self, D):
        # Apply asset transition
        Dtp1 = np.einsum("ij,ijk->ik", D, self.a_tmat)
        # Apply productivity transition
        Dtp1 = self.z_tmat.T @ Dtp1
        return Dtp1


# %% Solution class definition
class SimpleNKSolution(MetricObject):
    def __init__(self, grids, outcomes, vfunc, tmat=None):
        self.distance_criteria = ["vfunc"]
        self.grids = grids
        self.outcomes = outcomes
        self.vfunc = vfunc
        self.tmat = tmat


# %% Agent solver
@dataclass
class SimpleNKAgentSolver:
    solution_next: SimpleNKSolution
    crra: float
    beta: float
    beq_inten: float
    beq_shift: float
    LivPrb: float
    # Interest rates
    R: float
    # Income process
    prod_age_inter: float
    prod_pers_grid: np.ndarray
    prod_pers_trans: np.ndarray
    wage_rate: float
    working: bool
    a_grid: np.ndarray
    ls_transf: float
    tax_rate: float
    pension_mult: float

    # Auxiliary functions
    def dudc(self, c):
        return crra_marg(c, self.crra)

    def dudc_inv(self, c):
        return crra_marg_inv(c, self.crra)

    def u_inv(self, u):
        return crra_inv(u, self.crra)

    def beq_func(self, a):
        return beq_func(a, self.beq_inten, self.beq_shift, self.crra)

    def marg_beq_func(self, a):
        return marg_beq_func(a, self.beq_inten, self.beq_shift, self.crra)

    def income_func(self, z_t):
        # Other components of cash-on-hand
        if self.working:
            eff_hours = np.exp(self.prod_age_inter + z_t)
            ret_benef = 0.0
        else:
            eff_hours = 0.0
            ret_benef = np.exp(self.prod_age_inter + z_t) * self.pension_mult

        pret_inc = self.wage_rate * eff_hours + ret_benef
        tax = self.tax_rate * pret_inc
        net_transf = self.ls_transf + ret_benef - tax

        return pret_inc, tax, net_transf

    def solve(self):
        # Find grid dimensions
        n_zt = self.prod_pers_trans.shape[0]
        n_at = len(self.solution_next.grids["a_tm1"])

        # Find end-of-period value (dims zt, at)
        dgda = np.zeros((n_zt, n_at))
        # Survival (expected continuation)
        if self.LivPrb > 0.0:
            dgda += (
                self.beta
                * self.LivPrb
                * np.dot(self.prod_pers_trans, self.solution_next.vfunc["dvda"])
            )
        # Death (bequest)
        if self.LivPrb < 1.0:
            dgda += (1 - self.LivPrb) * self.marg_beq_func(
                self.solution_next.grids["a_tm1"][None, :]
            )

        # EGM inversion
        c_endo = self.dudc_inv(dgda)
        m_endo = self.solution_next.grids["a_tm1"][None, :] + c_endo

        # Evaluate the start-of-period outcomes and marginal value function

        # Reshape grids for broadcasting.
        # Order of dimensions is: z_t, a_tm1
        a_tm1 = self.a_grid[None, :]
        z_t = self.prod_pers_grid[:, None]

        # Assets after capital gains, "k_t"
        k_t = self.R * a_tm1

        # Various income and tax components
        pret_inc, tax, net_transf = self.income_func(z_t)
        # Income (after tax)
        income = pret_inc + self.ls_transf - tax

        # Cash on hand
        m_t = k_t + income

        # Interpolate consumption on exogenous grid
        c = interp_monotone(m_endo, c_endo, m_t)
        # Marginal value function on exogenous grid
        dvda = self.R * self.dudc(c)

        a_t = m_t - c

        # Outputs
        self.outputs = {
            "a_sop": a_tm1 + np.zeros_like(m_t),  # Broadcast
            "m": m_t,
            "c": c,
            "a": a_t,
            "a_dead": a_t * (1 - self.LivPrb),  # Assets of the dead
            "income": income + np.zeros_like(m_t),
            "net_transf": net_transf + np.zeros_like(m_t),
        }

        # Transition matrix to next period

        # Spread a_t mass on next period grid
        mass = mass_to_1d_grid_monotone(a_t, self.solution_next.grids["a_tm1"])
        # Expand with evolution of productivity
        tmat = NKtmat(mass, self.prod_pers_trans)

        # Construct and return solution
        return SimpleNKSolution(
            grids={"a_tm1": self.a_grid, "z_t": self.prod_pers_grid},
            outcomes=self.outputs,
            vfunc={"dvda": dvda},
            tmat=tmat,
        )


# %% Agent class definition
class SimpleNKAgent(AgentType):
    def __init__(self, cycles, tolerance=1e-10, **kwds):
        self.cycles = cycles
        self.pseudo_terminal = False
        self.tolerance = tolerance

        # Initialize parameters
        self.params = Parameters(**kwds)
        # Also save parameters in dictionary form as were called to initialize
        # the agent
        self.parameters = deepcopy(kwds)
        self.parameters.update({"cycles": cycles})

        # Set solver
        self.solve_one_period = make_one_period_oo_solver(SimpleNKAgentSolver)

        # Initialize other attributes
        self.update_solution_terminal()

    def pre_solve(self):
        # HARK expects a pre-solve method that we do not need in this case.
        return None

    def update_solution_terminal(self):
        # Creates a dummy terminal solution as the starting point for backward
        # iteration.

        if type(self.params["prod_pers_grid"]) == list:
            z_len = len(self.params["prod_pers_grid"][-1])
        else:
            z_len = len(self.params["prod_pers_grid"])

        if type(self.params["a_grid"]) == list:
            a_grid = self.params["a_grid"][-1]
        else:
            a_grid = self.params["a_grid"]

        # Create 'start-of-period' points that evaluate value and
        # marginal value
        # Assume agent gets no returns, some minimal income, and no transfers
        # and consumes everything.
        min_income = 1e-6
        m = np.zeros((z_len, len(a_grid))) + a_grid[None, :] + min_income

        self.solution_terminal = SimpleNKSolution(
            grids={"a_tm1": a_grid, "z_t": np.zeros(z_len)},
            outcomes={
                "a_sop": a_grid + np.zeros_like(m),
                "m": m,
                "c": m,
                "a": np.zeros_like(m),
                "income": np.zeros_like(m),
                "net_transf": np.zeros_like(m),
                "a_dead": np.zeros_like(m),
            },
            vfunc={"dvda": np.power(m, -self.params["crra"])},
        )

    def get_outcomes(self, outcomes):
        # Organize life cycle outcomes into a dictionary
        outcome_mesh = {
            out: [sol.outcomes[out] for sol in self.solution] for out in outcomes
        }

        return outcome_mesh

    def build_transitions(self, newborn_dstn=None):
        # Construct an object representing the transitions of a
        # population of agents.

        # Make survival probabilities a list if they are constant
        if isinstance(self.params["LivPrb"], list):
            surv_probs = self.params["LivPrb"]
        else:
            surv_probs = [self.params["LivPrb"]] * self.params._length

        if self.cycles == 0:
            # Infinite horizon
            self.transitions = DiscreteTransitions(
                living_tmats=[self.solution[0].tmat],
                surv_probs=surv_probs,
                life_cycle=False,
                newborn_dstn=newborn_dstn,
            )
        else:
            # Lifecycle
            self.transitions = DiscreteTransitions(
                living_tmats=[s.tmat for s in self.solution[:-1]],
                surv_probs=surv_probs,
                life_cycle=True,
                newborn_dstn=newborn_dstn,
            )


# %% Initial parametrization

# Preferences, constraints, and others
Rfree = 1.02
init_nk_params = {
    # Problem type (cycles = 0 means infinite horizon, cycles = 1 means lifecycle)
    "cycles": 1,
    # Preferences
    "crra": 3.0,
    "beta": 0.96,
    "LivPrb": [0.95, 0.9, 0.0],
    # Bequest
    "beq_inten": 2.0,
    "beq_shift": 1.5,
    # Interest rates.
    "R": Rfree,
    # Lump sum transfer
    "ls_transf": 0.0,
    # Pension multiplier
    "pension_mult": 1.0,
}

# Income
prod_pers_grid, _, prod_pers_trans = markov_rouwenhorst(rho=0.95, sigma=0.02, N=5)

# An initial dictionary with some parameters
init_income_params = {
    "wage_rate": 1.0,
    # Age intercepts
    "prod_age_inter": [1.0, 1.5, 2.0],
    # Persistent productivity
    "prod_pers_grid": prod_pers_grid,
    "prod_pers_trans": prod_pers_trans,
    # Working periods and retirement benefits
    "working": [True, True, False],
    # Tax
    "tax_rate": 0.0,
}

init_nk_params.update(init_income_params)

# Grids

# Wealth grid
a_grid = make_grid_exp_mult(ming=1e-4, maxg=100, ng=100, timestonest=2)
init_nk_params["a_grid"] = np.concatenate([np.array([0.0]), a_grid])
