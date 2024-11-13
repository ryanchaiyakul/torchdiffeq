import abc
import torch
from .solvers import FixedGridODESolver
from .misc import Perturb


class Optimizer(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def step(self):
        pass

    def zero(self):
        pass


class GradientDescent(Optimizer):

    def __init__(self, eta=1e-2, **unused_kwargs):
        self.eta = eta

    def step(self, y, gradG):
        return y - self.eta * gradG


class FletcherReeves(Optimizer):

    def __init__(self, eta=1e-2, **unused_kwargs):
        self.eta = eta
        self.p = None
        self.gradGm1 = None

    def step(self, y, gradG):
        if self.p is None:
            self.p = gradG
        else:
            beta = gradG.T * gradG / (self.gradGm1.T * self.gradGm1)
            self.p = gradG + beta * self.p
        self.gradGm1 = gradG
        return y - self.eta * gradG

    def zero(self):
        self.p = None
        self.gradGm1 = None


OPTIMIZERS = {'gd': GradientDescent,
              'fr': FletcherReeves}


class ProximalODESolver(metaclass=abc.ABCMeta):
    """
    Abstract class that contains useful functions to solve the inner minimization problem
    """

    def __init__(self, max_num_inner_steps=2**8 - 1, p_tol=1e-6, optimizer='gd', **unused_kwargs):
        self.max_iters = max_num_inner_steps
        self.tol = p_tol
        self.optim = OPTIMIZERS.get(optimizer)(**unused_kwargs)
        if self.optim is None:
            raise ValueError("Unknown optimizer {}".format(optimizer))

    @abc.abstractmethod
    def gradG(self, func, t0, dt, t1, y0, y):
        """
        Return the gradient of the expression you are trying to minimize
        """
        pass

    def in_tol(self, y, y_prev):
        return torch.norm(y - y_prev) < self.tol

    def _inner_step(self, func, t0, dt, t1, y0):
        """
        Perform iterative optimization steps until we reach the stopping condition (or max_iters)
        """
        y = y0
        self.optim.zero()
        for _ in range(self.max_iters):
            # To avoid side effects from in-place manipulations
            y_prev = y.clone()
            y = self.optim.step(y_prev, self.gradG(
                func, t0, dt, t1, y0, y))
            if self.in_tol(y, y_prev):
                break
        return y - y0, self._func(func, t0, y)

    def _func(self, func, t0, y):
        """
        Helper to call a perturb object with the proper arguments
        """
        return func(t0, y, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)


class BackwardEuler(FixedGridODESolver, ProximalODESolver):
    order = 1

    def __init__(self, **unused_kwargs):
        ProximalODESolver.__init__(self, **unused_kwargs)
        FixedGridODESolver.__init__(self, **unused_kwargs)

    def _step_func(self, func, t0, dt, t1, y0):
        return self._inner_step(
            func, t0, dt, t1, y0)

    def gradG(self, func, t0, dt, t1, y0, y):
        return (y - y0) / dt - self._func(func, t0, y)
