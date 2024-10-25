# Policy Search
import numpy as np
import random
from abc import abstractmethod
from typing import Any, Callable
from scipy.stats import rv_continuous, multivariate_normal
from src.ExactSolutionMethods import MDP, MDPSolutionMethod
from src.OnlinePlanning import rollout


class MonteCarloPolicyEvaluation:
    def __init__(self, P: MDP, b: np.ndarray, d: int, m: int):
        self.P = P  # problem
        self.b = b  # initial state distribution
        self.d = d
        self.m = m  # number of samples

    def evaluate_policy(self, policy: Callable[[Any], Any]) -> float:
        state = random.choices(self.P.S, weights=self.b)[0]
        return np.mean([rollout(self.P, state, policy, self.d) for _ in range(self.m)])

    def evaluate_parametrized_policy(self, policy: Callable[[np.ndarray, Any], Any],
                                     theta: np.ndarray) -> float:
        return self.evaluate_policy(lambda s: policy(theta, s))


class PolicySearchMethod(MDPSolutionMethod):
    @abstractmethod
    def optimize(self, policy: Callable[[np.ndarray, Any], Any],
                 U: MonteCarloPolicyEvaluation) -> np.ndarray:
        pass


class SearchDistributionMethod(MDPSolutionMethod):
    @abstractmethod
    def optimize(self, policy: Callable[[np.ndarray, Any], Any],
                 U: MonteCarloPolicyEvaluation) -> rv_continuous:
        pass


class HookeJeevesPolicySearch(PolicySearchMethod):
    def __init__(self, theta: np.ndarray, alpha: float, c: float, epsilon: float):
        self.theta = theta
        self.alpha = alpha  # step size
        self.c = c  # step size reduction factor
        self.epsilon = epsilon  # termination step size

    def optimize(self, policy: Callable[[np.ndarray, Any], Any],
                 U: MonteCarloPolicyEvaluation) -> np.ndarray:
        theta, theta_prime = self.theta.copy(), np.zeros(self.theta.shape)
        u, n = U(policy, theta), len(theta)
        while self.alpha > self.epsilon:
            np.copyto(dst=theta_prime, src=theta)
            best = (0, 0, u)  # (i, sgn, u)
            for i in range(n):
                for sgn in [-1, 1]:
                    theta_prime[i] = theta[i] + sgn * self.alpha
                    u_prime = U(policy, theta_prime)
                    if u_prime > best[2]:
                        
