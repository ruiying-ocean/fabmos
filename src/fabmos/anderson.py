"""Anderson acceleration of a fixed-point iteration x <- g(x).

Each new iterate is the combination of previous ones that minimizes the linearized
residual g(x) - x. This follows the formulation of Walker & Ni (2011,
https://doi.org/10.1137/10078356X), with the damping of Anderson (1965).
"""

import numpy as np


class AndersonAccelerator:
    def __init__(self, m: int = 50, beta: float = 1.0):
        """
        Args:
            m: number of previous iterates to combine. 0 gives the plain iteration.
                At its peak, memory use is about 4 * m times the size of the state.
            beta: damping. 1 takes the full step, smaller positive values a shorter
                step in the same direction.
        """
        if m < 0:
            raise ValueError(f"m must not be negative, but is {m}")
        if not 0.0 < beta <= 1.0:
            raise ValueError(f"beta must be above 0 and at most 1, but is {beta}")
        self.m = m
        self.beta = beta
        self._g = None
        self._f = None
        self._dG: list[np.ndarray] = []
        self._dF: list[np.ndarray] = []

    def update(self, x: np.ndarray, gx: np.ndarray) -> np.ndarray:
        """Return the next iterate.

        Args:
            x: current iterate. This must be the previous result of this method,
                or a state derived from it.
            gx: its image g(x)
        """
        f = gx - x
        if self._f is not None and self.m > 0:
            self._dG.append(gx - self._g)
            self._dF.append(f - self._f)
            if len(self._dF) > self.m:
                del self._dG[0], self._dF[0]
        self._g = gx.copy()
        self._f = f

        x_next = gx - (1.0 - self.beta) * f
        if self._dF:
            # lstsq returns the minimum-norm solution, so that differences that are
            # zero or linearly dependent (as near convergence) do no harm
            dF = np.column_stack(self._dF)
            gamma = np.linalg.lstsq(dF, f, rcond=None)[0]
            if self.beta != 1.0:
                x_next += (1.0 - self.beta) * (dF @ gamma)
            del dF
            for weight, dg in zip(gamma, self._dG):
                x_next -= weight * dg
        return x_next
