import diffusion as df
import networkx as nx
import numpy as np


def activation_probability_generator(i):
    return .95  # 95%


def activation_callback(n):
    # n["ap"] -> probability of activation
    # normal.random < n["ap"] ?
    return np.random.randint(2, size=1)[0]


def infection_probability_generator(i):
    return 1.0  # 100% - if i's neighbour is infected, i gets infected


def infection_callback(u, v):
    # infection probability of node v is denoted by v["ip"]
    # some random? normal.rand < v["ip"] ?
    return True


def starting_nodes_callback(G):
    # take first k of nodes with the biggest rank? betweenness?
    G.nodes[0]["infected"] = True


def post_stage_callback(G, stage):
    print("Stage num: %d" % stage)
    # dump G.nodes[i]["infected"]?
    # calculate coverage?


if __name__ == "__main__":
    size = 300
    G = nx.random_geometric_graph(size, 0.125)
    diffusion = df.Diffusion(G, activation_probability_generator, activation_callback, infection_probability_generator,
                             infection_callback, starting_nodes_callback, post_stage_callback)
    df.visualisation(diffusion, 10)
