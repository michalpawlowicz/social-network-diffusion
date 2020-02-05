import diffusion as df
import networkx as nx
import numpy as np
import random


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


def starting_nodes_callback(G, start_method, n, min_distance):
    if start_method == 'degree':
        starting_nodes_by_degree(G, n, min_distance)
    elif start_method == 'betweenness':
        starting_nodes_by_betweenness_centrality(G, n, min_distance)
    elif start_method == 'closeness':
        starting_nodes_by_closeness_centrality(G, n, min_distance)
    elif start_method == 'random':
        starting_nodes_random(G, n)


def starting_nodes_random(G, n):
    random_nodes = random.sample(range(len(G.nodes)), n)
    for node_idx in random_nodes:
        G.nodes[node_idx]["infected"] = True


def starting_nodes_by_closeness_centrality(G, n, min_distance):
    # closeness centrality of a node is calculated as the sum of the length of the shortest paths between the node and all other nodes in the graph.
    # infect first n nodes with the biggest closeness centrality
    degrees = [(n, b) for n, b in nx.algorithms.closeness_centrality(G).items()]
    degrees.sort(key=lambda node: node[1], reverse=True)
    sorted_nodes = [best_node[0] for best_node in degrees]
    start_nodes = select_nodes_with_distance(G, sorted_nodes, n, min_distance)
    for node_idx in start_nodes:
        G.nodes[node_idx]["infected"] = True


def starting_nodes_by_betweenness_centrality(G, n, min_distance):
    # Betweenness centrality measures how important a node is to the shortest paths through the network
    # infect first n nodes with the biggest betweenness centrality
    degrees = [(n, b) for n, b in nx.algorithms.betweenness_centrality(G).items()]
    degrees.sort(key=lambda node: node[1], reverse=True)
    sorted_nodes = [best_node[0] for best_node in degrees]
    start_nodes = select_nodes_with_distance(G, sorted_nodes, n, min_distance)
    for node_idx in start_nodes:
        G.nodes[node_idx]["infected"] = True


def starting_nodes_by_degree(G, n, min_distance):
    # infect first n nodes with the biggest degree
    degrees = list(G.degree)
    degrees.sort(key=lambda node: node[1], reverse=True)
    sorted_nodes = [best_node[0] for best_node in degrees]
    start_nodes = select_nodes_with_distance(G, sorted_nodes, n, min_distance)
    for node_idx in start_nodes:
        G.nodes[node_idx]["infected"] = True


def select_nodes_with_distance(G, nodes, n, min_distance):
    filtered_starting_nodes = [nodes[0]]
    if n == 1:
        return filtered_starting_nodes
    else:
        for distance in reversed(range(1, min_distance+1)):
            for node in nodes[1:]:
                add = True
                for starting_node in filtered_starting_nodes:
                    if nx.shortest_path_length(G, node, starting_node) < distance:
                        add = False
                        break
                if add:
                    filtered_starting_nodes.append(node)
                if len(filtered_starting_nodes) >= n:
                    return filtered_starting_nodes


def post_stage_callback(G, stage):
    print("Stage num: %d" % stage)
    # dump G.nodes[i]["infected"]?
    # calculate coverage?


if __name__ == "__main__":
    graph_size = 250  # graph nodes number
    mu_activ, sigma_activ = 0.3, 0.05  # mean and standard deviation for activation_probability
    mu_infect, sigma_infect = 0.3, 0.05  # mean and standard deviation for infection_probability
    stages = 100  # stages for visualisation
    goal = 0.90  # goal infected graph percentage for performance check
    stages_perfor = 300  # maximum number of stages to reach the goal
    attempts = 100  # number of attempts taken into account for performance checking
    plateau_tolerance = 200  # maximum number of diffusions without progress before being treated as failure
    start_nodes_num = 5  # number of nodes that we want to infect in the beginning to start propagation
    min_distance = 3  # minimal distance between selected starting nodes
    start_method = 'degree'  # name of starting point selection method AVAILABLE: 'degree', 'betweenness', 'closeness'
    G = nx.random_geometric_graph(graph_size, 0.1)  # specified graph generation
    average_stages_finished, failures_percentage = df.check_graph_performance(G, goal, attempts, stages_perfor, plateau_tolerance,
                                                            mu_activ, sigma_activ, mu_infect, sigma_infect, start_nodes_num, min_distance, start_method,
                                                            activation_probability_generator, activation_callback,
                                                            infection_probability_generator, infection_callback,
                                                            starting_nodes_callback)
    print("\nAverage stages number needed to achieve goal percentage infections {}".format(average_stages_finished))
    print("Percentage of attempts that didn't achieve  goal {}".format(failures_percentage))
