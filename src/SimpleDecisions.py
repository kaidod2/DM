# Simple Decisions
import numpy as np
from src.Representation import Variable, Assignment, BayesianNetwork, assignments
from src.Inference import DiscreteInferenceMethod


class SimpleProblem:
    def __init__(self, bn: BayesianNetwork,
                 chance_varas: list[Variable],
                 decision_vars: list[Variable],
                 utility_vars: list[Variable],
                 utilities: dict[str, np.ndarray]):
        self.bn = bn
        self.chance_varas = chance_varas
        self.decision_vars = decision_vars
        self.utility_vars = utility_vars
        self.utilities = utilities

    def solve(self, evidence: Assignment, M: DiscreteInferenceMethod) -> tuple[Assignment, float]:
        query = [var.name for var in self.utility_vars]

        def U(a):
            return np.sum([self.utilities[uname][a[uname]] for uname in query])

        best_a, best_u = None, -np.inf
        for assignment in assignments(self.decision_vars):
            evidence = Assignment(evidence | assignment)
            phi = M.infer(self.bn, query, evidence)
            u = np.sum([p * U(a) for a, p in phi.table.items()])
            if u > best_u:
                best_a, best_u = assignment, u
        return best_a, best_u

    def value_to_information(self, query: list[str], evidence: Assignment, M: DiscreteInferenceMethod) -> float:
        phi = M.infer(self.bn, query, evidence)
        voi = -(self.solve(evidence, M)[1])
        query_vars = [var for var in self.chance_varas if var.name in query]
        for o_p in assignments(query_vars):
            o_o_p = Assignment(evidence | o_p)
            p = phi.table[o_p]
            voi += p * (self.solve(o_o_p, M)[1])
        return voi
