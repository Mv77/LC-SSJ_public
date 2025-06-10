from dataclasses import dataclass

import numpy as np


# %% Class that represents a transition matrix
class TransitionMatrix:
    """
    Representation of a transition matrix that transforms
    (grid_t) to (grid_t+1).
    Post multiplication must take an array with the shape of grid_{t+1} and transform it to the size of grid_t.
    Pre multiplication must take an array with the shape of grid_t and transform it to the size of grid_{t+1}.
    """

    def __init__(self):
        pass

    def premult(self, D):
        """
        Generic operation that advances a distribution to the next time period.
        D should have the shape of the current grid (grid_t) and the method should return
        a distribution with the shape of the next grid (grid_{t+1}).
        """
        raise NotImplementedError(
            "This method should be implemented in subclasses of TransitionMatrix."
        )

    def postmult(self, outcomes):
        """
        Generic operation that takes expectations of outcomes in the next time period conditional
        on current states.
        Outcomes should have shape (..., grid_{t+1}) and the method should return a tensor of shape
        (..., grid_t).
        """
        raise NotImplementedError(
            "This method should be implemented in subclasses of TransitionMatrix."
        )


# %% Class and methods that facilitates simulating populations in discretized
# state spaces
@dataclass
class DiscreteTransitions:
    """
    Class to facilitate simulating transitions of populations in discretized state spaces,
    supporting both life-cycle and infinite-horizon models.
    The class assumes that:
     - Death is exogenous and independent of every state.
     - Agents that die are replaced by newborns.
     - Newborns draw their state from a distribution that is constant over time.

    Parameters
    ----------
    living_tmats : list[TransitionMatrix]
        List of transition matrices conditional on survival for each period (life-cycle) or
        a single matrix (infinite-horizon).
    surv_probs : list[float]
        List of survival probabilities for each period (life-cycle) or a single probability (infinite-horizon).
    life_cycle : bool
        If True, use life-cycle mode; otherwise, use infinite-horizon mode.
    newborn_dstn : np.array
        Distribution of newborns (initial distribution).
    """

    living_tmats: list
    surv_probs: list
    life_cycle: bool
    newborn_dstn: np.array

    def __post_init__(self):
        """
        Initialize the DiscreteTransitions object and check parameter consistency.
        """
        if self.life_cycle:
            self.T = len(self.living_tmats) + 1
            if len(self.surv_probs) != (self.T - 1):
                raise ValueError(
                    f"surv_probs must have length {len(self.living_tmats)}."
                )
        else:
            self.T = 1

    def iterate_dstn_forward(self, dstn_init):
        """
        Propagate a distribution forward one period.

        Parameters
        ----------
        dstn_init : list[np.ndarray]
            Initial distribution to propagate. Must be a list of length T (life-cycle) or a
            single array (infinite-horizon).

        Returns
        -------
        list[np.array]
            The propagated distribution(s).
        """
        if self.life_cycle:
            return _iterate_dstn_forward_lc(
                dstn_init, self.living_tmats, self.surv_probs, self.newborn_dstn
            )
        else:
            return [
                _iterate_dstn_forward_ih(
                    dstn_init[0],
                    self.living_tmats[0],
                    self.surv_probs[0],
                    self.newborn_dstn,
                )
            ]

    def find_conditional_age_dsnt(self, dstn_init):
        """
        Given a distribution of agents over states for the first period of life,
        find the distribution of agents over states in every age conditional on
        their survival.

        Parameters
        ----------
        dstn_init : list[np.ndarray]
            Initial distribution.

        Returns
        -------
        list[np.array]
            List of distributions by age (life-cycle) or a single-element list (infinite-horizon).
        """
        if self.life_cycle:
            return _find_conditional_age_dsnt(dstn_init, self.living_tmats)
        else:
            return dstn_init.copy()

    def find_steady_state_dstn(self, **kwargs):
        """
        Find the steady-state distribution.

        Parameters
        ----------
        **kwargs : dict
            Additional arguments for infinite-horizon steady-state solver.
            Examples include:
                - max_iter: int, maximum number of iterations (default: 10000).
                - tol: float, tolerance for convergence (default: 1e-10).
                - normalize_every: int, frequency of normalization (default: 100).
                - dstn_init: np.array, initial distribution (default: None).

        Returns
        -------
        list[np.array]
            List of steady-state distributions by age (life-cycle) or a single-element list (infinite-horizon).
        """
        if self.life_cycle:
            if kwargs:
                raise ValueError(
                    "kwargs are not used in the life cycle version of find_steady_state_dstn."
                )
            return _find_steady_state_dstn_lc(
                self.surv_probs, self.newborn_dstn, self.living_tmats
            )
        else:
            return [
                _find_steady_state_dstn_ih(
                    self.newborn_dstn,
                    self.living_tmats[0],
                    self.surv_probs[0],
                    **kwargs,
                )
            ]


# Life cycle methods
def _iterate_dstn_forward_lc(dstn_init, living_tmats, surv_probs, newborn_dstn):
    new_dstn = [newborn_dstn.copy()]
    dead_mass = 0.0
    for i, (d0, tmat) in enumerate(zip(dstn_init, living_tmats)):
        new_dstn.append(tmat.premult(surv_probs[i] * d0))
        dead_mass += (1.0 - surv_probs[i]) * np.sum(d0)
    dead_mass += np.sum(dstn_init[-1])
    new_dstn[0] *= dead_mass

    return new_dstn


def _find_conditional_age_dsnt(dstn_init, living_tmats):
    dstns = [dstn_init.copy()]
    for tmat in living_tmats:
        dstns.append(tmat.premult(dstns[-1]))
    return dstns


def _find_steady_state_dstn_lc(surv_probs, newborn_dstn, living_tmats):
    ss_age_mass = np.empty(len(surv_probs) + 1)
    ss_age_mass[0] = 1.0
    for i in range(1, len(ss_age_mass)):
        ss_age_mass[i] = ss_age_mass[i - 1] * surv_probs[i - 1]
    ss_age_mass /= np.sum(ss_age_mass)
    age_dstns = _find_conditional_age_dsnt(newborn_dstn, living_tmats)
    return [age_dstn * age_mass for age_dstn, age_mass in zip(age_dstns, ss_age_mass)]


# Infinite horizon methods
def _iterate_dstn_forward_ih(dstn_init, living_tmat, surv_prob, newborn_dstn):
    dead_mass = 1.0 - surv_prob
    new_dstn = surv_prob * living_tmat.premult(dstn_init)
    new_dstn += dead_mass * newborn_dstn

    return new_dstn


def _find_steady_state_dstn_ih(
    newborn_dstn,
    living_tmat,
    surv_prob,
    max_iter=10000,
    tol=1e-10,
    normalize_every=100,
    dstn_init=None,
):
    if dstn_init is None:
        dstn = newborn_dstn
    else:
        dstn = dstn_init
    go = True
    i = 0
    while go:
        new_dstn = _iterate_dstn_forward_ih(dstn, living_tmat, surv_prob, newborn_dstn)
        if np.linalg.norm(new_dstn - dstn) < tol:
            go = False
        dstn = new_dstn
        i += 1
        if i > max_iter:
            go = False
        # Renormalize every given number of iterations
        if i % normalize_every == 0:
            dstn /= np.sum(dstn)

    # Return as list just for compatibility with LC methods that return
    # a list of age dstns
    return dstn
