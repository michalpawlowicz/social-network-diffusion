import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from tqdm import tqdm


class Diffusion:
    def __init__(self, graph, mu_activ, sigma_activ, mu_infect, sigma_infect, ap_callback: callable, activation_callback: callable, infection_probability_callback: callable,
                 infection_callback: callable, starting_nodes_callback: callable, post_stage_callback=None):
        """
        :param graph: Network represented as Graph
        :param ap_callback: Function taking integer as argument which represents id of node and returns activation
                            probability of this node. Used as generator during initialization
        :param activation_callback Function taking Node as argument, return if node is active in given stage
        :param infection_probability_callback: Function taking integer as argument which represents id of node
                                                and returns infection probability of this node. Used as generator
                                                during initialization
        :param infection_callback: Function taking two parameters u, v of type Node, and returns True,
                                    if v is infected by u, False otherwise.
                                    Called every time if infection was possible between u and v
        :param starting_nodes_callback: Function taking graph as parameter, and sets up entry nodes
        :param post_stage_callback: if provided, Diffusion will call it after every stage passing Graph reference
                                    and stage number. Might be used to calculate diffusion percentage of every stage
                                    or to reseed if diffusion suppressed
        """
        self.G = graph
        nx.set_node_attributes(self.G, values=True, name="active")
        nx.set_node_attributes(self.G, values=False, name="infected")

        # infection state have to be updated after iteration to avoid races
        # this attribute stores infection state of pending iteration
        nx.set_node_attributes(self.G, values=False, name="infected_copy")
        for i in self.G:
            self.G.nodes[i]["ap"] = ap_callback(mu_activ, sigma_activ)  # activation probability
            self.G.nodes[i]["ip"] = infection_probability_callback(mu_infect, sigma_infect)  # infection probability
        self.infectionCallback = infection_callback
        starting_nodes_callback(self.G)

        self.post_stage_callback = post_stage_callback

        self.activation_callback = activation_callback;

    def _update_activation_state(self):
        for i in self.G.nodes:
            if not self.G.nodes[i]["infected"]:
                self.G.nodes[i]["active"] = self.activation_callback(self.G.nodes[i]) #

    def _propagate(self):
        it = nx.bfs_successors(self.G, 0)
        for v in it:
            if self.G.nodes[v[0]]["infected"]:
                for u in v[1]:
                    if self.G.nodes[u]["active"]:
                        if self.infectionCallback(self.G.nodes[v[0]], self.G.nodes[u]):
                            self.G.nodes[u]["infected_copy"] = True

        # iteration has ended, infection state can be updated
        for v in self.G.nodes:
            n = self.G.nodes[v]
            if not n["infected"] and n["infected_copy"]:
                n["infected"] = True

    def diffuse(self, k):
        self._update_activation_state()
        self._propagate()
        if self.post_stage_callback is not None:
            self.post_stage_callback(self.G, k)

    def diffusion(self, count):
        for i in count:
            self.diffuse(i)


def color(node):
    if node["infected"]:
        return "red"
    elif node["active"]:
        return "green"
    else:
        return "gray"


def update_visualisation(num, layout, G, ax, D):
    ax.clear()
    random_colors = [color(G.nodes[i]) for i in G.nodes]
    nx.draw(G, pos=layout, node_color=random_colors, ax=ax, node_size=30)
    D.diffuse(num)


def visualisation(diffusion, stages):
    fig, ax = plt.subplots(figsize=(10, 10))
    layout = nx.spring_layout(diffusion.G)
    ani = animation.FuncAnimation(fig, update_visualisation, frames=stages, repeat=False, fargs=(layout, diffusion.G, ax, diffusion))
    plt.show()
    # ani.save("test.gif", writer='imagemagick', fps=1)


def check_graph_performance(G, goal, attempts, stages, plateau_tolerance, mu_activ, sigma_activ, mu_infect, sigma_infect,
                          activation_probability_generator, activation_callback, infection_probability_generator,
                          infection_callback, starting_nodes_callback):
    failures = 0
    stages_finished = []
    for _ in tqdm(range(attempts), desc="All attempts progress"):
        diffusion = Diffusion(G, mu_activ, sigma_activ, mu_infect, sigma_infect, activation_probability_generator,
                             activation_callback, infection_probability_generator, infection_callback,
                             starting_nodes_callback)
        result = _check_diffusion_on_goal(diffusion, goal, stages, plateau_tolerance)
        if result is None:
            failures += 1
        else:
            stages_finished.append(result)
    failures_percentage = failures / attempts
    average_stages_finished = np.mean(stages_finished)
    return average_stages_finished, failures_percentage


def _check_diffusion_on_goal(diffusion, goal, stages, plateau_tolerance):
    plateau_amount = 0
    last_infected_percentage = 0
    for stage in range(stages):
        diffusion.diffuse(stage)
        infected_percentage = _get_infected_percentage(diffusion)
        plateau_amount = _check_diffusion_progress(infected_percentage, last_infected_percentage, plateau_amount)
        last_infected_percentage = infected_percentage
        if plateau_amount > plateau_tolerance:
            return None
        if infected_percentage >= goal:
            return stage


def _check_diffusion_progress(infected_percentage, last_infected_percentage, plateau_amount):
    if infected_percentage > last_infected_percentage:
        plateau_amount = 0
    else:
        plateau_amount += 1
    return plateau_amount


def _get_infected_percentage(diffusion):
    nodes_all = diffusion.G.number_of_nodes()
    nodes_infected = len([node for node in diffusion.G.nodes(data='infected') if node[1] is True])
    return nodes_infected / nodes_all
