import diffusion as df
import networkx as nx
import numpy as np


def activation_probability_generator(mu, sigma):
    return np.random.normal(mu, sigma)  # normal distribution with 'mu' mean and 'sigma' standard deviation


def activation_callback(n):
    odds = np.random.uniform()
    if odds <= n["ap"]:
        return 1
    else:
        return 0


def infection_probability_generator(mu, sigma):
    return np.random.normal(mu, sigma)  # normal distribution with 'mu' mean and 'sigma' standard deviation


def infection_callback(u, v):
    odds = np.random.uniform()
    if odds <= v["ip"]:
        return True
    else:
        return False


def starting_nodes_callback(G, start_method, n):
    if start_method == 'degree':
        starting_nodes_by_degree(G, n)


def starting_nodes_by_degree(G, n):
    # infect first n nodes with the biggest degree
    degrees = list(G.degree)
    degrees.sort(key=lambda node: node[1], reverse=True)
    start_nodes = [best_node[0] for best_node in degrees[:n]]
    for node_idx in start_nodes:
        G.nodes[node_idx]["infected"] = True


def post_stage_callback(G, stage):
    print("Stage num: %d" % stage)
    # dump G.nodes[i]["infected"]?
    # calculate coverage?


if __name__ == "__main__":
    graph_size = 200  # graph nodes number
    mu_activ, sigma_activ = 0.3, 0.05  # mean and standard deviation for activation_probability
    mu_infect, sigma_infect = 0.3, 0.05  # mean and standard deviation for infection_probability
    stages = 100  # stages for visualisation
    goal = 0.90  # goal infected graph percentage for performance check
    stages_perfor = 200  # maximum number of stages to reach the goal
    attempts = 100  # number of attempts taken into account for performance checking
    plateau_tolerance = 200  # maximum number of diffusions without progress before being treated as failure
    start_nodes_num = 5  # number of nodes that we want to infect in the beginning to start propagation
    start_method = 'degree'  # name of starting point selection method AVAILABLE: 'degree'
    G = nx.random_geometric_graph(graph_size, 0.125)    # specified graph generation
    diffusion = df.Diffusion(G, mu_activ, sigma_activ, mu_infect, sigma_infect, start_nodes_num, start_method, activation_probability_generator,
                             activation_callback, infection_probability_generator, infection_callback,
                             starting_nodes_callback, post_stage_callback)
    df.visualisation(diffusion=diffusion, stages=stages)
    average_stages_finished, failures_percentage = df.check_graph_performance(G, goal, attempts, stages_perfor, plateau_tolerance,
                                                                            mu_activ, sigma_activ, mu_infect, sigma_infect, start_nodes_num, start_method,
                                                                            activation_probability_generator, activation_callback,
                                                                            infection_probability_generator, infection_callback,
                                                                            starting_nodes_callback)
    print("\nAverage stages number needed to achieve goal percentage infections {}".format(average_stages_finished))
    print("Percentage of attempts that didn't achieve  goal {}".format(failures_percentage))

