# Parameter Learning
import networkx as nx
import numpy as np

from scipy.stats import norm
from typing import Callable

from src.Representation import Variable


def statistics(variables: list[Variable], graph: nx.DiGraph, data: np.ndarray) -> list[np.ndarray]:
    n = len(variables)
    r = np.array([var.r for var in variables])
    q = np.array([int(np.prod([r[j] for j in graph.predecessors(i)])) for i in range(n)])
    M = [np.zeros((q[i], r[i])) for i in range(n)]
    for o in data.T:
        for i in range(n):
            k = o[i]
            parents = list(graph.predecessors(i))
            j = 0
            if len(parents) != 0:
                j = np.ravel_multi_index(o[parents], r[parents])
            M[i][j, k] += 1.0
    return M


def prior(variables: list[Variable], graph: nx.DiGraph) -> list[np.ndarray]:
    n = len(variables)
    r = [var.r for var in variables]
    q = np.array([int(np.prod([r[j] for j in graph.predecessors(i)])) for i in range(n)])
    return [np.ones((q[i], r[i])) for i in range(n)]


def gaussian_kernel(b: float) -> Callable[[float], float]:
    return lambda x: norm.pdf(x, loc=0, scale=b)


def kernel_density_estimate(kernel: Callable[[float | np.ndarray], float], observation: np.ndarray) -> Callable:
    return lambda x: np.mean([kernel(x - o) for o in observation])
