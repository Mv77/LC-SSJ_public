from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


# %% Class that represents a transition matrix
class TransitionMatrix(ABC):
    """
    Transition matrix that maps a state distribution at t into one at t + 1.

    `premult` advances distributions forward in time. `postmult` pushes
    next-period outcomes back to current states to form conditional
    expectations.
    """

    @abstractmethod
    def premult(self, D):
        """
        Advance a distribution from the current grid to the next grid.
        """

    @abstractmethod
    def postmult(self, outcomes):
        """
        Map next-period outcomes to current-state expectations.
        """

    def advance(self, D):
        """
        Readable alias for `premult`.
        """
        return self.premult(D)

    def expectat(self, outcomes):
        """
        Readable alias for `postmult`.
        """
        return self.postmult(outcomes)


# %% Class and methods that facilitates simulating populations in discretized
# state spaces
@dataclass
class DiscreteTransitions:
    """
    Population transition wrapper for either a life-cycle or an
    infinite-horizon model.

    The class assumes that newborns draw their state from a distribution
    that is constant over time. This distribution must add up to 1.0.

    Parameters
    ----------
    tmats : list[TransitionMatrix]
        Transition matrices that account for death.
    life_cycle : bool
        Whether the object represents a life-cycle model.
    nb_dstn : np.array
        Distribution of newborns.
    """

    tmats: list
    life_cycle: bool
    nb_dstn: np.array

    def __post_init__(self):
        """
        Initialize the DiscreteTransitions object and check parameter consistency.
        """
        if self.life_cycle:
            self.T = len(self.tmats) + 1
        else:
            self.T = 1
            if len(self.tmats) != 1:
                raise ValueError(
                    "Infinite-horizon transitions require exactly one transition matrix."
                )

        # Check that nb_dstn is a valid distribution
        if not np.isclose(self.nb_dstn.sum(), 1.0):
            raise ValueError("nb_dstn must sum to 1.0.")

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
            Updated age distributions. In infinite-horizon mode this is a
            single-element list.
        """
        if self.life_cycle:
            return _iterate_dstn_forward_lc(dstn_init, self.tmats, self.nb_dstn)
        else:
            return [
                _iterate_dstn_forward_ih(
                    dstn_init,
                    self.tmats,
                    self.nb_dstn,
                )
            ]

    def find_conditional_age_dstn(self, dstn_init):
        """
        Starting from the first-age distribution, build the age profile of
        conditional distributions.

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
            return _find_conditional_age_dstn(dstn_init, self.tmats)
        else:
            return [dstn_init.copy()]

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
            List of steady-state distributions by age. In infinite-horizon mode
            this is a single-element list.
        """
        if self.life_cycle:
            if kwargs:
                raise ValueError(
                    "kwargs are not used in the life cycle version of find_steady_state_dstn."
                )
            return _find_steady_state_dstn_lc(self.tmats, self.nb_dstn)
        else:
            return _find_steady_state_dstn_ih(
                self.tmats,
                self.nb_dstn,
                **kwargs,
            )


# Life cycle methods
def _iterate_dstn_forward_lc(dstn_init, tmats, nb_dstn):
    new_dstn = [nb_dstn.copy()]
    for i, (d0, tmat) in enumerate(zip(dstn_init, tmats)):
        new_dstn.append(tmat.advance(d0))
    return new_dstn


def _find_conditional_age_dstn(dstn_init, tmats):
    dstns = [dstn_init.copy()]
    for i, tmat in enumerate(tmats):
        dstns.append(tmat.advance(dstns[-1]))
    return dstns


def _find_steady_state_dstn_lc(tmats, nb_dstn):
    return _find_conditional_age_dstn(nb_dstn, tmats)


# Infinite horizon methods
def _iterate_dstn_forward_ih(dstn_init, tmats, nb_dstn):
    new_dstn = tmats[0].advance(dstn_init) + nb_dstn
    return new_dstn


def _find_steady_state_dstn_ih(
    tmats,
    nb_dstn,
    max_iter=10000,
    tol=1e-16,
    dstn_init=None,
):
    if dstn_init is None:
        dstn = nb_dstn
    else:
        dstn = dstn_init
    go = True
    i = 0
    size = dstn.size
    while go:
        new_dstn = _iterate_dstn_forward_ih(dstn, tmats, nb_dstn)
        if np.linalg.norm(new_dstn - dstn) / size < tol:
            go = False
        dstn = new_dstn
        i += 1
        if i > max_iter:
            go = False

    # Return as list just for compatibility with LC methods that return
    # a list of age dstns
    return [dstn]
