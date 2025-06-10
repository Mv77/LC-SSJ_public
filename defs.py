import numpy as np
from HARK.Calibration.life_tables.us_ssa.SSATools import parse_ssa_life_table
from HARK.utilities import make_grid_exp_mult
from sequence_jacobian.utilities.discretize import markov_rouwenhorst

# %% Jac parameters
T = 300
dx = 1e-4


# %% Paths depend on whether we are in a notebook or not
def is_interactive():
    """Return ``True`` when running in an interactive session."""
    import __main__ as main

    return not hasattr(main, "__file__")


# Root folder
if is_interactive():
    root_dir = ".."
else:
    root_dir = "."

# %% Baseline parametrization (household)

# Grid sizes
a_grid_size = 50

# Age parameters
birth_age = 26
death_age = 100
age_ret = 65
age_vec = np.arange(birth_age, death_age + 1)
# A boolean vector of whether the agent is working or retired
working_vec = np.arange(birth_age, death_age + 1) <= age_ret

# Preferences, constraints, and others
params = {
    # Problem type (cycles = 0 means infinite horizon, cycles = 1 means lifecycle in HARK)
    "cycles": 1,
    "T_cycle": death_age - birth_age + 1,
    # Preferences
    "crra": 2.0,
    "beta": 0.98,
    # Bequest
    "beq_inten": 0.0,
    "beq_shift": 1e-5,
    # Financials
    "R": 1.02,
    # Lump sum transfers
    "ls_transf": 0.0,
    # Pension multiplier
    "pension_mult": 1.0,
    # Taxation
    "tax_rate": 0.3,
}

# Asset grid
a_grid = make_grid_exp_mult(ming=1e-4, maxg=500, ng=a_grid_size, timestonest=2)
a_grid = np.concatenate([np.array([0.0]), a_grid])
params["a_grid"] = a_grid

# %% Length of life and mortality

# We need survival probabilities only up to death_age-1, because survival
# probability at death_age is 1.
liv_prb = parse_ssa_life_table(
    female=False, cross_sec=True, year=2004, min_age=birth_age, max_age=death_age - 1
)
liv_prb = liv_prb + [0.0]

assert len(liv_prb) == (death_age - birth_age + 1)

# Update survival probabilities
params.update({"LivPrb": liv_prb})

# %% Income process

# Age fixed effects from log-income polynomial estimated on the 2019 scf
ln_y_poly_coefs = np.array(
    [2.4817156, -0.5989584, 0.8934862, -0.2480151, 0.0264144, -0.0009903]
)
age_mat = np.stack(
    [np.power(age_vec / 10, i) for i in range(len(ln_y_poly_coefs))], axis=1
)
income_log_int = np.dot(age_mat, ln_y_poly_coefs)

# Productivity shocks
prod_pers = 0.95
prod_shk_sd = 0.2
prod_points = 7
prod_grid, prod_ss, prod_tmat = markov_rouwenhorst(
    rho=prod_pers, sigma=prod_shk_sd, N=prod_points
)

# Normalize income by the mean of exp(intercepts) across ages
income_base = np.mean(np.exp(income_log_int))
norm_income_log_intercepts = [i - np.log(income_base) for i in income_log_int]

# Income process: constant grid that evolves during working life but becomes constant in retirment.
prod_pers_grid = np.log(prod_grid)
prod_pers_trans = [
    prod_tmat if age < age_ret else np.eye(len(prod_grid)) for age in age_vec
]

# An initial dictionary with some parameters
income_params = {
    "wage_rate": 1.0,
    # Age intercepts
    "prod_age_inter": norm_income_log_intercepts,
    # Persistent productivity
    "prod_pers_grid": prod_pers_grid,
    "prod_pers_trans": prod_pers_trans,
    # Working ages and retirement benefits
    "working": [bool(x) for x in working_vec],
}
params.update(income_params)

# %% Newborn distribution

# Find steady state work productivity dstn
prod_ss_work = prod_ss

# Everyone starts at minimum assets and the SS distribution of productivity
nb_dstn = np.zeros_like(a_grid)
nb_dstn[0] = 1.0
# Apply z distribution
nb_dstn = nb_dstn * prod_ss_work[:, None]
