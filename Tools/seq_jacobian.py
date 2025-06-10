from copy import deepcopy

import numpy as np
from numba import jit

from Tools.transition_matrices import DiscreteTransitions

# %% Misc


# Recursive map function for lists and dicts
def rec_map(fn, x):
    """Recursively apply ``fn`` to all elements of lists or dictionaries."""
    if isinstance(x, dict):
        return {k: rec_map(fn, v) for k, v in x.items()}
    elif isinstance(x, list):
        return [rec_map(fn, v) for v in x]
    else:
        return fn(x)


# %% Functions to produce shocked outcomes and transitions


# Infinite horizon version
def shock_input_ih(
    agent,
    shk_param,
    outcome_fns,
    newborn_dstn,
    horizon=200,
    dx=1e-3,
):
    """Solve a temporary finite-horizon version of an IH agent with a shock."""
    # %% Step 1: Set up a finite horizon clone of the agent
    params = deepcopy(agent.params.to_dict())
    params["T_cycle"] = horizon
    params["cycles"] = 1
    shock_date = horizon - 1

    # Make the relevant parameter time-varying and shock it
    params[shk_param] = [params[shk_param]] * horizon
    params[shk_param][shock_date] += dx

    # Create the agent
    fh_agent = type(agent)(**params)

    # %% Step 2: Solve the agent starting from the IH steady state solution
    # Finite horizon method requires
    fh_agent.solve(from_solution=agent.solution[0])

    # %% Step 3: Find the transition matrices and outcomes
    fh_agent.build_transitions(newborn_dstn=newborn_dstn)
    shocked_living_tmats = fh_agent.transitions.living_tmats
    shocked_responses = fh_agent.get_outcomes(outcome_fns)

    # %% Step 4: Organize outputs

    # Reverse the outcomes
    for key, item in shocked_responses.items():
        shocked_responses[key] = np.stack([x for x in reversed(item[:-1])])
        # Drop the last period, it is the SS solution passed as terminal

    # Reverse the transition matrices (they do not have transitions associated with terminal solution)
    shocked_living_tmats = reversed(shocked_living_tmats)

    # Form transition matrices as ih matrix objects
    shocked_transitions = [
        DiscreteTransitions(
            living_tmats=[lt],
            surv_probs=[agent.params["LivPrb"]],
            life_cycle=False,
            newborn_dstn=newborn_dstn,
        )
        for lt in shocked_living_tmats
    ]

    return shocked_transitions, shocked_responses


def shock_input_lc(ss_agent, shk_param, dx, s, outcome_fns):
    """Shock a parameter in a life-cycle model and resolve future ages."""
    # %% Create the shocked agent

    # Make a base parameter list
    params = deepcopy(ss_agent.parameters)

    # Make sure that the parameter we will shock is time-varying
    if type(params[shk_param]) != list:
        params[shk_param] = (s + 1) * [params[shk_param]]

    # Trim all time-varying parameters
    for key, item in params.items():
        if type(item) == list:
            params[key] = item[: (s + 1)]

    # Shock the parameter
    params[shk_param][s] += dx

    # Set length of life cycle
    params["T_cycle"] = s + 1  # 0,1,...,s = s+1 ages

    # Solve from the ss solution of the following age
    solve_from = ss_agent.solution[s + 1]

    # Create and solve the agent
    shocked_agent = type(ss_agent)(**params)
    shocked_agent.solve(from_solution=solve_from)

    # Get the transition matrices and outcomes
    shocked_agent.build_transitions()
    shocked_responses = shocked_agent.get_outcomes(outcome_fns)

    return shocked_agent.transitions.living_tmats, shocked_responses


# %% Infinite horizon Jacobians
def fake_news_jacobian_ih(
    agent,
    shk_param,
    outcome_fns,
    newborn_dstn,
    horizon=200,
    dx=1e-3,
    verbose=False,
    steady_state=None,
):
    """Compute Jacobians in an infinite-horizon model using fake news shocks."""
    if verbose:
        print("Step 1: steady state")

    # Extract information from the agent:
    # 1. The transition matrix
    # 2. The outcomes of interest evaluated on the state mesh
    # 3. The steady state distribution
    ss_transition = agent.transitions
    if steady_state is None:
        ss_points = agent.get_outcomes(outcome_fns)
        D_ss = agent.transitions.find_steady_state_dstn()[0]
    else:
        ss_points = steady_state["points"]
        D_ss = steady_state["dstn"]

    if verbose:
        print("Step 2: shocked responses and distributions")

    # Find the "shocked" responses and distributions
    shocked_transitions, shocked_points = shock_input_ih(
        agent,
        shk_param,
        outcome_fns,
        newborn_dstn,
        horizon,
        dx,
    )

    # Stack outcomes. Dimensions are (outcome, [states])
    out_names = list(ss_points.keys())
    nout = len(out_names)
    ss_points = list(map(lambda x: np.stack(x, axis=0), zip(*ss_points.values())))[0]

    # Stack shocked outcomes and flatten their state dimension. Dimensions are (outcome, s, [states])
    shocked_points = np.stack(
        [x.reshape((horizon, -1)) for x in shocked_points.values()]
    )

    if verbose:
        print("Step 3: Jacobians")

    # Calculate the steady state value of the outcomes
    ss_outcomes = np.array([np.sum(y_ss * D_ss) for y_ss in ss_points])

    # %% Step 2: expectation vectors

    # The expectation of policy vectors conditional on death is the same
    # for each gridpoint
    death_exp = (1 - ss_transition.surv_probs[0]) * np.array(
        [np.sum(ss_transition.newborn_dstn * y_ss) for y_ss in ss_points]
    )
    # Broadcast to the shape of policy vectors
    death_exp = death_exp.reshape(
        tuple(x if xi == 0 else 1 for xi, x in enumerate(ss_points.shape))
    )

    # Initialize
    curly_E = [ss_points]
    # Fill in
    for t in range(1, horizon):
        curly_E.append(
            death_exp
            + ss_transition.surv_probs[0]
            * ss_transition.living_tmats[0].postmult(curly_E[-1])
        )

    # After getting expectation vectors there is no gain from keeping states
    # represented in dimensions. Flatten them to make expectations faster
    curly_E = rec_map(lambda x: x.reshape((nout, -1)), curly_E)

    # %% Step 1: Changes in outcomes and distributions

    # Outcomes
    curly_Y = (np.dot(shocked_points, D_ss.flatten()) - ss_outcomes[:, None]) / dx

    # Distributions
    shocked_D1s = [x.iterate_dstn_forward([D_ss])[0] for x in shocked_transitions]
    curly_D = (
        np.stack([(d1_shk - D_ss) for d1_shk in shocked_D1s]).reshape(horizon, -1) / dx
    )

    # %% Step 3: Fake news matrix
    curly_F = {}
    curly_F = np.zeros((nout, horizon, horizon))
    curly_F[:, 0, :] += curly_Y
    for i in range(1, horizon):
        curly_F[:, i, :] += np.dot(curly_E[i - 1], curly_D.T)

    # %% Step 4: Jacobian
    curly_J = curly_F.copy()
    for t in range(1, horizon):
        curly_J[:, 1:, t] += curly_J[:, :-1, t - 1]

    curly_J = {key: curly_J[i] for i, key in enumerate(out_names)}
    curly_F = {key: curly_F[i] for i, key in enumerate(out_names)}

    return curly_J, curly_F


# %% Life cycle Jacobians
def fake_news_jacobian_lc(
    agent,
    shk_param,
    outcome_fns,
    horizon=200,
    dx=1e-3,
    verbose=False,
    steady_state=None,
):
    """Compute Jacobians in a life-cycle model using fake news shocks."""
    # Get the number of ages
    A = agent.params._length

    # Extract information from the agent:
    # 1. The transition matrix
    # 2. The outcomes of interest evaluated on the state mesh
    # 3. The steady state distribution
    ss_transition = agent.transitions
    if steady_state is None:
        ss_points = agent.get_outcomes(outcome_fns)
        D_ss = agent.transitions.find_steady_state_dstn()
    else:
        ss_points = steady_state["points"]
        D_ss = steady_state["dstn"]

    livprb = agent.params["LivPrb"]

    # Get expectation vectors
    if verbose:
        print("Step 2: Expectation vectors")

    # Stack outcomes. List indexes age, np array is [outcome, grid]
    out_names = list(ss_points.keys())
    ss_points = list(map(lambda x: np.stack(x, axis=0), zip(*ss_points.values())))
    e_vecs = _get_expectation_vectors(
        ss_points, ss_transition.living_tmats, agent.params["LivPrb"]
    )
    # The only thing left to do with evecs is multiply them by shocked distributions.
    # This is much faster if they are flat arrays.
    nouts = len(out_names)
    e_vecs = rec_map(lambda x: x.reshape((nouts, -1)), e_vecs)
    # Also, all the e_vecs of a given age (with varying t) have the same shape.
    # Stack them to be able to multiply by all of them at once
    e_vecs = [np.stack(x) for x in e_vecs]
    # Index of list is age a, first dimension of array is time t, second dimension is the outcome, third is the grid

    # %% Step 3: Solve shocked agents and fill fake news matrices
    if verbose:
        print("Step 3: shocked solutions and fake news matrices")

    # Initialize the fake news matrices
    Fn_mats = np.zeros((len(out_names), A, A, A))

    for k in reversed(range(A)):
        if verbose:
            print(f"Shocked Age {k}/{A - 1}")

        shk_tmats, shk_responses = shock_input_lc(agent, shk_param, dx, k, outcome_fns)

        # Update first row elements of the fake news matrices
        for out_no, key in enumerate(out_names):
            for a in range(k + 1):
                Fn_mats[out_no, a, 0, k - a] += np.sum(
                    (shk_responses[key][a] - ss_points[a][out_no, ...]) * D_ss[a]
                )

        # Update the rest of the elements
        for a in range(k + 1):
            # Flatten distribution for easier sums
            dD1 = (livprb[a] * shk_tmats[a].premult(D_ss[a]) - D_ss[a + 1]).flatten()
            update_Fn_mats(Fn_mats, e_vecs[a + 1], dD1, A, a, k)

    # Normalize everything by the shock size and pad with 0s to
    # the horizon
    Fn_mats /= dx
    Fn_mats = np.pad(Fn_mats, ((0, 0), (0, 0), (0, horizon - A), (0, horizon - A)))

    # Form jacobians
    Jacs = Fn_mats.copy()
    for t in range(1, horizon):
        Jacs[:, :, 1:, t] += Jacs[:, :, :-1, t - 1]

    curly_J = {key: Jacs[i] for i, key in enumerate(out_names)}
    curly_F = {key: Fn_mats[i] for i, key in enumerate(out_names)}

    return curly_J, curly_F


@jit(nopython=True)
def update_Fn_mats(Fn_mats, evecs, dD1, A, a, k):
    """Update fake news matrices with contributions from one period."""
    # Dimensions:
    # Fn_mats: (n_out, A, A, A)
    # evecs : (T, n_out, G)
    # dD1   : (G,)
    n_out = Fn_mats.shape[0]
    G = dD1.shape[0]

    for oi in range(n_out):
        for t in range(1, A - a):
            # Compute dot(evecs[t-1, oi, :], dD1) manually for Numba compatibility
            s = 0.0
            for g in range(G):
                s += evecs[t - 1, oi, g] * dD1[g]
            Fn_mats[oi, a + t, t, k - a] += s


def iterate_exp_vector(livprb, living_tmat, evec):
    """Propagate an expectation vector forward one period."""
    return livprb * living_tmat.postmult(evec)


def _get_expectation_vectors(points, living_tmats, surv_probs):
    """Compute expectation vectors for each age in a life-cycle model."""
    A = len(points)
    evecs = [[p.copy()] for p in points]
    for t in range(1, A):
        for a in range(A - t):
            evecs[a].append(
                iterate_exp_vector(surv_probs[a], living_tmats[a], evecs[a + 1][-1])
            )
    return evecs


# %% Actual Jacobian calculation


# Wrapper function
def fake_news_jacobian(
    agent,
    shk_param,
    outcome_fns,
    newborn_dstn,
    horizon=200,
    dx=1e-3,
    verbose=False,
    steady_state=None,
):
    """
    Calculate the Jacobian matrix for a given agent model.

    This function determines whether the model is life-cycle or infinite horizon
    based on the `cycles` attribute of the agent. It then calls the appropriate
    function to compute the Jacobian matrix.

    Parameters:
        agent: The agent object containing the model to analyze.
        shk_param: The parameter to shock in the model.
        outcome_fns: A list of functions to compute outcomes of interest.
        newborn_dstn: The distribution of newborn agents.
        horizon: The time horizon for the analysis (default is 200).
        dx: The size of the shock to apply to the parameter (default is 1e-3).
        verbose: Whether to print detailed progress information (default is False).
        steady_state: Precomputed steady-state information (optional).

    Returns:
        A tuple containing:
        - curly_J: The Jacobian matrix.
        - curly_F: The fake news matrix.
        - ss_outcomes: The steady-state outcomes.
    """

    # Determine whether the model is life-cycle or infinite horizon
    # The `cycles` attribute specifies the type of model:
    # `cycles == 0` indicates an infinite horizon model,
    # `cycles == 1` indicates a life-cycle model.
    if agent.cycles == 0:
        infinite_horizon = True
    elif agent.cycles == 1:
        infinite_horizon = False
    else:
        raise ValueError(
            f"Model is not life-cycle or infinite horizon. agent.cycles = {agent.cycles}"
        )

    # Call the appropriate function
    if infinite_horizon:
        return fake_news_jacobian_ih(
            agent=agent,
            shk_param=shk_param,
            outcome_fns=outcome_fns,
            newborn_dstn=newborn_dstn,
            horizon=horizon,
            dx=dx,
            verbose=verbose,
            steady_state=steady_state,
        )
    else:
        return fake_news_jacobian_lc(
            agent=agent,
            shk_param=shk_param,
            outcome_fns=outcome_fns,
            horizon=horizon,
            dx=dx,
            verbose=verbose,
            steady_state=steady_state,
        )
