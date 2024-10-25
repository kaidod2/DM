import numpy as np
from abc import abstractmethod
from typing import Any, Callable

from src.ExactSolutionMethods import MDP, MDPSolutionMethod, ValueFunctionPolicy


class OnlinePlanningMethod(MDPSolutionMethod):
    @abstractmethod
    def __call__(self, s: Any) -> Any:
        pass


def rollout(P: MDP, s: Any, policy: Callable[[Any], Any], d: int) -> float:
    ret = 0.0
    for t in range(d):
        a = policy(s)
        s, r = P.randstep(s, a)
        ret += (P.gamma ** t) * r
    return ret


class RolloutLookahead(OnlinePlanningMethod):
    def __init__(self, P: MDP, policy: Callable[[Any], Any], d: int):
        self.P = P  # problem
        self.policy = policy  # rollout policy
        self.d = d  # depth

    def __call__(self, s: Any) -> Any:
        def U(s):  return rollout(self.P, s, self.policy, self.d)

        return self.P.greedy(U, s)[0]


def forward_search(P: MDP, s: Any, d: int,
                   U: Callable[[Any], float]) -> tuple[Any, float]:
    if d <= 0:
        return None, U(s),
    best_a, best_u = (None, -np.inf)

    def U_prime(s):
        return forward_search(P, s, d - 1, U)[1]

    for a in P.A:
        u = P.lookahead(U_prime, s, a)
        if u > best_u:
            best_a, best_u = a, u
    return best_a, best_u


class ForwardSearch(OnlinePlanningMethod):
    def __init__(self, P: MDP, d: int, U: Callable[[Any], float]):
        self.P = P  # problem
        self.d = d  # depth
        self.U = U  # value function at depth d

    def __call__(self, s: Any) -> Any:
        return forward_search(self.P, s, self.d, self.U)[0]

