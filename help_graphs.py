
import numpy as np
import networkx as nx
import random

from itertools import combinations
from fractions import Fraction
from concurrent.futures import ProcessPoolExecutor
################################

#function to load graphs file generated with GAP
def load_graphs_from_txt(filename):
    graphs = []
    with open(filename) as f:
        while line := f.readline():
            n, m = map(int, line.split())
            G = nx.Graph()
            G.add_nodes_from(range(n))
            for _ in range(m):
                u, v = map(int, f.readline().split())
                G.add_edge(u, v)
            graphs.append(G)
    return graphs

##################################

#only use for graphs without isolated vertices
def normalized_laplacian(G):
    A = nx.to_numpy_array(G)
    degrees = A.sum(axis=1)
    sqrtinvs = [1/np.sqrt(x) for x in degrees]
    D = np.diag(sqrtinvs)
    L = np.diag(degrees) - A
    return D @ L @ D

def eigenvals_lap(G):
    L = normalized_laplacian(G)
    eigenvals = np.linalg.eigvalsh(L)
    return eigenvals[1:len(L)//2]


#only use for connected graphs
def spectral_gap(G):
    eigenvals = eigenvals_lap(G)
    return eigenvals[0]

################################

#only use for regular graphs
def graph_data(N, E):
    adj = [[] for _ in range(N)]
    for u, v in E:
        adj[u].append(v)
        adj[v].append(u)
    d = len(adj[0])
    if any(len(a) != d for a in adj):
        raise ValueError("Graph must be regular")
    return adj, d

def boundary_size(E, S):
    return sum((u in S) != (v in S) for u, v in E)

def best_variation(N, E, adj, d, S):
    S = set(S)
    b = boundary_size(E, S)
    best_num, best_den = b, d * len(S)
    best_set = S
    for r in (1, 2):
        for T in combinations(range(N), r):
            new_size = len(S) + sum(-1 if v in S else 1 for v in T)
            if not 1 <= new_size <= N // 2:
                continue
            Tset = set(T)
            affected = set()
            for u in T:
                for v in adj[u]:
                    affected.add((min(u, v), max(u, v)))
            b2 = b
            for u, v in affected:
                before = (u in S) != (v in S)
                after = ((u in S) ^ (u in Tset)) != ((v in S) ^ (v in Tset))
                b2 += after - before
            den2 = d * new_size
            if b2 * best_den < best_num * den2:
                best_num, best_den = b2, den2
                best_set = S.symmetric_difference(Tset)
    return Fraction(best_num, best_den), best_set

def good_cheeger_value(G):
    N = G.number_of_nodes()
    E = list(G.edges())
    adj, d = graph_data(N, E)
    best_value, best_set = None, None
    for _ in range(100):
        k = random.randint(1, N // 2)
        S = set(random.sample(range(N), k))
        for _ in range(200):
            value, S2 = best_variation(N, E, adj, d, S)
            if S2 == S:
                break
            S = S2
        if best_value is None or value < best_value:
            best_value, best_set = value, S
    return best_value, best_set

#computing cheeger constant parallel
def _cheeger_worker(G):
    return ChegA.good_cheeger_value(G)

def compute_cheeger_values(graphs):
    with ProcessPoolExecutor(max_workers=24) as executor:
        results = list(executor.map(_cheeger_worker, graphs))
    return ([value for value, S in results],[S for value, S in results])

##################################

def value_for_witness(G, witness):
    E = list(G.edges())
    degrees = {deg for _, deg in G.degree()}
    if len(degrees) != 1:
        raise ValueError("Graph must be regular")
    d = degrees.pop()
    delta = ChegA.boundary_size(E, witness)
    return Fraction(delta, d * len(witness))

def verify_graph_triple(G, cheeger, witness):
    return value_for_witness(G, witness) == cheeger

def verify_graph_triples(graphs, cheegers, witnesses):
    if not (len(graphs) == len(cheegers) == len(witnesses)):
        return False
    for G, cheeger, witness in zip(graphs, cheegers, witnesses):
        if not verify_graph_triple(G, cheeger, witness):
            return False
    return True
