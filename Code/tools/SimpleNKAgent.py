from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from HARK.core import AgentType, Parameters, make_one_period_oo_solver
from HARK.metric import MetricObject
from HARK.utilities import make_grid_exp_mult
from numba import float64, guvectorize, jit
from sequence_jacobian.grids import markov_rouwenhorst
# Project-defined tools
from tools.transition_matrices import DiscreteTransitions, TransitionMatrix


# %% Interpolators
@guvectorize(
    [(float64[:], float64[:], float64[:], float64[:])],
    "(n),(n),(p)->(p)",
    nopython=True,
)
def interp_monotone(m, c, m_new, c_new):
    """Monotone interpolation of consumption policy."""
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
    """Distribute mass from arbitrary points onto a 1D grid."""
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


# %% Transition matrix class for the model
class NKtmat(TransitionMatrix):
    """Transition matrix for the SimpleNK model."""

    def __init__(self, a_tmat, z_tmat, liv_prb):
        super().__init__()
        self.a_tmat = a_tmat
        self.z_tmat = z_tmat
        self.liv_prb = liv_prb

    def postmult(self, out):
        # 2d state space: (z, a)
        exp = np.einsum("...mk,jm", out, self.z_tmat)
        exp *= self.liv_prb
        exp = np.einsum("...jk,jlk->...jl", exp, self.a_tmat)
        return exp

    def premult(self, D):
        # 2d state space: (z, a)
        Dtp1 = np.einsum("ij,ijk->ik", D, self.a_tmat)
        Dtp1 *= self.liv_prb
        Dtp1 = np.einsum("jm,jk->mk", self.z_tmat, Dtp1)
        return Dtp1


# %% Solution class definition
class SimpleNKSolution(MetricObject):
    """Container for the solution of a single period."""

    def __init__(self, grids, outcomes, vfunc, tmat=None):
        self.distance_criteria = ["vfunc"]
        self.grids = grids
        self.outcomes = outcomes
        self.vfunc = vfunc
        self.tmat = tmat


# %% Agent solver
@dataclass
class SimpleNKAgentSolver:
    """One-period solver for the SimpleNKAgent."""

    solution_next: SimpleNKSolution
    # Preferences
    crra: float
    beta: float
    # Survival
    LivPrb: float
    # Interest rates
    R: float
    # Asset grid
    a_grid: np.ndarray
    # Income process
    prod_age_inter: float
    prod_pers_grid: np.ndarray
    prod_pers_trans: np.ndarray
    wage_rate: float
    working: bool
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

    def income_func(self, z_t):
        prod = np.exp(self.prod_age_inter + z_t)
        # Other components of cash-on-hand
        if self.working:
            # Hours normalized to 1, so effective hours are just productivity.
            eff_hours = prod
            ret_benef = np.zeros_like(prod)
        else:
            eff_hours = np.zeros_like(prod)
            ret_benef = prod * self.pension_mult

        pret_inc = self.wage_rate * eff_hours + ret_benef
        tax = self.tax_rate * pret_inc
        net_transf = self.ls_transf + ret_benef - tax

        return eff_hours, ret_benef, pret_inc, tax, net_transf

    def solve(self):
        """Solve one period of the household problem via EGM."""
        # Find grid dimensions
        n_zt = self.prod_pers_trans.shape[0]
        n_atm1 = len(self.a_grid)
        n_at = len(self.solution_next.grids["a_tm1"])

        # Find end-of-period value dims (zt, at) integrating over
        # z_{t+1} and discounting.
        dgda = self.beta * np.einsum(
            "jm,ma->ja", self.prod_pers_trans, self.solution_next.vfunc["dvda"]
        )

        # EGM inversion dims (zt, at)
        c_endo = self.dudc_inv(dgda)
        m_endo = self.solution_next.grids["a_tm1"][None, :] + c_endo

        # Evaluate outcomes on exogenous grid. dims (zt, a_{t-1})
        a_tm1 = self.a_grid[None, :]
        z_t = self.prod_pers_grid[:, None]

        # Initialize start-of-period marginal value function (zt, a_{t-1})
        dvda = np.zeros((n_zt, n_atm1))

        # Assets after capital gains, "k_t"
        k_t = self.R * a_tm1

        # Income and taxes
        eff_hours, ret_benef, pret_inc, tax, net_transf = self.income_func(z_t)

        # Retirement benefits are in both pret_inc and net_transf, so subtract once.
        income = (pret_inc - ret_benef) + net_transf

        # Cash on hand
        m_t = k_t + income

        # Interpolate consumption on exogenous grid
        c = interp_monotone(m_endo, c_endo, m_t)

        # Add continuation value of surviving agents to marginal value function
        if self.LivPrb > 0.0:
            dvda += self.LivPrb * self.R * self.dudc(c)

        # Assets at the end of the period
        a_t = m_t - c

        # Outputs. They have to be adjusted for survival before integration.
        self.outputs = {
            "B_out": (1.0 - self.LivPrb) * a_tm1,  # Bequest of the dying
            "C": self.LivPrb * c,  # Consumption of the living
            "A": self.LivPrb * a_t,  # End-of-period assets of the living
            "N": self.LivPrb * eff_hours,  # Effective labor supply of the living
            "P": self.LivPrb * np.ones_like(c),  # Surviving population
            "T": -1.0 * self.LivPrb * net_transf,  # Net taxes paid by the living
            "Income": self.LivPrb * income,  # Income of the living
            "Pwork": self.LivPrb
            * np.array([[[self.working]]]),  # Indicator for working
        }

        # Transition matrix to next period

        # Spread a_t mass on next period grid
        mass = mass_to_1d_grid_monotone(a_t, self.solution_next.grids["a_tm1"])
        # Expand with evolution of productivity
        tmat = NKtmat(mass, self.prod_pers_trans, self.LivPrb)

        # Construct and return solution
        return SimpleNKSolution(
            grids={"a_tm1": self.a_grid, "z_t": self.prod_pers_grid},
            outcomes=self.outputs,
            vfunc={"dvda": dvda},
            tmat=tmat,
        )


# %% Agent class definition
class SimpleNKAgent(AgentType):
    """Lifecycle agent used in the examples and tests."""

    def __init__(self, cycles, tolerance=1e-10, nb_dstn=None, **kwds):
        """Initialize the agent with model parameters."""
        self.cycles = cycles
        self.pseudo_terminal = False
        self.tolerance = tolerance
        self.nb_dstn = nb_dstn

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
        """Placeholder for compatibility with HARK's API."""
        return None

    def update_solution_terminal(self):
        """Create a dummy terminal solution used to start backward iteration."""

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
                "B_out": np.zeros_like(m),  # Bequest of the dying
                "C": m,  # Consumption of the living
                "A": np.zeros_like(m),  # End-of-period assets of the living
                "N": np.zeros_like(m),  # Effective labor supply of the living
                "P": np.zeros_like(m),  # Surviving population
                "T": np.zeros_like(m),  # Net taxes paid by the living
                "Income": np.zeros_like(m),  # Income of the living
                "Pwork": np.zeros_like(m),  # Indicator for working
            },
            vfunc={"dvda": np.power(m, -self.params["crra"])},
        )

    def get_outcomes(self, outcomes):
        """Return requested outcomes for each period as a dictionary."""
        # Organize life cycle outcomes into a dictionary
        outcome_mesh = {
            out: [sol.outcomes[out] for sol in self.solution] for out in outcomes
        }

        return outcome_mesh

    def build_transitions(self):
        """Create a ``DiscreteTransitions`` object for the agent."""
        # Construct an object representing the transitions of a
        # population of agents.

        # Check that nb_dstn integrates to 1
        if self.nb_dstn is not None:
            if abs(np.sum(self.nb_dstn) - 1.0) > 1e-8:
                raise ValueError("nb_dstn does not integrate to 1.")
        else:
            raise ValueError("nb_dstn has not been set. Cannot build transitions.")

        nb_mass = self.nb_dstn / np.sum(self.nb_dstn)

        if self.cycles == 0:
            # Infinite horizon
            self.transitions = DiscreteTransitions(
                tmats=[self.solution[0].tmat],
                life_cycle=False,
                nb_dstn=nb_mass,
            )
        else:
            # Lifecycle
            self.transitions = DiscreteTransitions(
                tmats=[s.tmat for s in self.solution[:-1]],
                life_cycle=True,
                nb_dstn=nb_mass,
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
    "prod_fe": np.array([0.0]),
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
