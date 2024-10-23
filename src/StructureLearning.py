# Structure Learning
import itertools
import networkx as nx
import numpy as np

from abc import ABC, abstractmethod
from scipy.special import loggamma

from src.Representation import Variable
from src.ParameterLearning import prior, statistics


def bayesian_score_component(M: np.ndarray, alpha: np.ndarray) -> float:
    alpha_0 = np.sum(alpha, axis=1)
    p = np.sum(loggamma(alpha + M))
    p -= np.sum(loggamma(alpha))
    p += np.sum(loggamma(alpha_0))
    p -= np.sum(loggamma(alpha_0 + np.sum(M, axis=1)))
    return p


def bayesian_score(variables: list[Variable],
                   graph: nx.DiGraph, data: np.ndarray) -> float:
    n = len(variables)
    M = statistics(variables, graph, data)
    alpha = prior(variables, graph)
    return np.sum([bayesian_score_component(M[i], alpha[i]) for i in range(n)])


class DirectedGraphSearchMethod(ABC):
    @abstractmethod
    def fit(self, variables: list[Variable], data: np.ndarray) -> nx.DiGraph:
        pass


class K2Search(DirectedGraphSearchMethod):
    def __init__(self, ordering: list[int]):
        self.ordering = ordering

    def fit(self, variables: list[Variable], data: np.ndarray) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_nodes_from(range(len(variables)))
        for k, i in enumerate(self.ordering[1:]):
            y = bayesian_score(variables, graph, data)
            while True:
                y_best, j_best = -np.inf, 0
                for j in self.ordering[:k]:
                    if not graph.has_edge(j, i):
                        graph.add_edge(j, i)
                        y_prime = bayesian_score(variables, graph, data)
                        if y_prime > y_best:
                            y_best, j_best = y_prime, j
                        graph.remove_edge(j, i)

                if y_best > y:
                    y = y_best
                    graph.add_edge(j_best, i)
                else:
                    break
        return graph

