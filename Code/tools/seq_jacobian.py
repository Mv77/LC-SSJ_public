from copy import deepcopy

import numpy as np
from numba import jit

# %% Misc


# Recursive map function for lists and dicts
def rec_map(fn, x):
    if isinstance(x, dict):
        return {k: rec_map(fn, v) for k, v in x.items()}
    elif isinstance(x, list):
        return [rec_map(fn, v) for v in x]
    else:
        return fn(x)


def _stack_outcomes_by_age(outcome_points):
    """
    Build per-age outcome stacks while allowing broadcast-compatible shapes.

    For each age, arrays from different outcomes are broadcast to a common
    shape and then stacked as [outcome, ...grid_dims...].
    """
    stacked = []
    for age_points in zip(*outcome_points.values()):
        arrays = [np.asarray(x) for x in age_points]
        broadcasted = np.broadcast_arrays(*arrays)
        stacked.append(np.stack(broadcasted, axis=0))
    return stacked


# %% Functions to produce shocked outcomes and transitions


def shock_input_lc(ss_agent, shk_param, dx, s, outcomes):
    # %% Create the shocked agent

    # Make a base parameter list
    params = deepcopy(ss_agent.parameters)
    params["nb_dstn"] = ss_agent.nb_dstn

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
    shocked_responses = shocked_agent.get_outcomes(outcomes)

    return shocked_agent.transitions.tmats, shocked_responses


# %% Life cycle Jacobians
def fake_news_jacobian_lc(
    agent,
    shk_param,
    outcomes,
    horizon=200,
    dx=1e-3,
    verbose=False,
    steady_state=None,
):
    # Get the number of ages
    A = agent.params._length

    # Extract information from the agent:
    # 1. The transition matrix
    # 2. The outcomes of interest evaluated on the state mesh
    # 3. The steady state distribution
    ss_transition = agent.transitions
    if steady_state is None:
        ss_points = agent.get_outcomes(outcomes)
        D_ss = agent.transitions.find_steady_state_dstn()
    else:
        ss_points = steady_state["points"]
        D_ss = steady_state["dstn"]

    # Get expectation vectors
    if verbose:
        print("Step 2: Expectation vectors")

    # Stack outcomes. List indexes age, np array is [outcome, grid]
    out_names = list(ss_points.keys())
    ss_points = _stack_outcomes_by_age(ss_points)
    e_vecs = _get_expectation_vectors(ss_points, ss_transition.tmats)
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

        shk_tmats, shk_responses = shock_input_lc(
            agent,
            shk_param,
            dx,
            k,
            outcomes,
        )

        # Update first row elements of the fake news matrices
        for out_no, key in enumerate(out_names):
            for a in range(k + 1):
                Fn_mats[out_no, a, 0, k - a] += np.sum(
                    (shk_responses[key][a] - ss_points[a][out_no, ...]) * D_ss[a]
                )

        # Update the rest of the elements
        for a in range(k + 1):
            # Flatten distribution for easier sums
            dD1 = (shk_tmats[a].premult(D_ss[a]) - D_ss[a + 1]).flatten()
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
    """
    Fn_mats: (n_out, A, A, A)
    evecs : (T, n_out, G)
    dD1   : (G,)
    """
    n_out = Fn_mats.shape[0]
    G = dD1.shape[0]

    for oi in range(n_out):
        for t in range(1, A - a):
            # compute dot(evecs[t-1, oi, :], dD1) by hand
            s = 0.0
            for g in range(G):
                s += evecs[t - 1, oi, g] * dD1[g]
            Fn_mats[oi, a + t, t, k - a] += s


def _get_expectation_vectors(points, tmats):
    A = len(points)
    evecs = [[p.copy()] for p in points]
    for t in range(1, A):
        for a in range(A - t):
            evecs[a].append(tmats[a].postmult(evecs[a + 1][-1]))
    return evecs


# %% Actual Jacobian calculation


# Wrapper function
def fake_news_jacobian(
    agent,
    shk_param,
    outcomes,
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
        raise NotImplementedError(
            "Fake news Jacobian for infinite horizon not implemented."
        )
    else:
        return fake_news_jacobian_lc(
            agent=agent,
            shk_param=shk_param,
            outcomes=outcomes,
            horizon=horizon,
            dx=dx,
            verbose=verbose,
            steady_state=steady_state,
        )
