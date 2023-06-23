import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

# Referenced: https://github.com/javieryu/nn_distributed_training

def get_metropolis_weights(graph):
    "Return a sparse matrix weighting each agents' contributions: more emphasis on self"

    N = graph.number_of_nodes()
    W = torch.zeros((N, N))

    L = nx.laplacian_matrix(graph)
    degrees = [L[i, i] for i in range(N)]

    for i in range(N):
        for j in range(N):
            if graph.has_edge(i, j) and i != j:
                W[i, j] = 1.0 / (max(degrees[i], degrees[j]) + 1.0)

    for i in range(N):
        W[i, i] = 1.0 - torch.sum(W[i, :])

    return W

def get_neighbor_weights(graph):
    "Return a sparse matrix weighting each agents' contributions equally"

    N = graph.number_of_nodes()
    W = torch.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i == j or graph.has_edge(i, j):
                W[i, j] = 1.0

    return F.normalize(W, p=1, dim=1)

def generate_graph(graph_config):
    N = graph_config.num_nodes
    if graph_config.type == "wheel":
        graph = nx.wheel_graph(N)
    elif graph_config.type == "cycle":
        graph = nx.cycle_graph(N)
    elif graph_config.type == "complete":
        graph = nx.complete_graph(N)
    elif graph_config.type == "input" and graph_config.graph is not None:
        graph = nx.from_numpy_array(np.array(graph_config.graph))
        if not nx.is_connected(graph):
            raise NameError("Specified graph is not connected.")
    elif graph_config.type == "random":
        # Attempt to make a random graph until it is connected
        graph = nx.erdos_renyi_graph(N, graph_config.p)
        for _ in range(graph_config.gen_attempts):
            if nx.is_connected(graph):
                break
            else:
                graph = nx.erdos_renyi_graph(N, graph_config.p)

        if not nx.is_connected(graph):
            raise NameError(
                "A connected random graph could not be generated,"
                " increase p or gen_attempts."
            )
    else:
        raise NameError("Unknown communication graph type.")
    return graph