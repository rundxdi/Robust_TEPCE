
import mpinput as mp
import networkx as nx
import numpy as np
from networkx import algorithms
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

filename = 'pglib-opf-master/az_2020_case892.m'

bus_data = np.array([])
gen_data = np.array([])
branch_data = np.array([])
bus_data,gen_data,branch_data = mp.load_data(filename)


cap = 1
graph = nx.Graph()
mp.encode_graph(graph, bus_data, gen_data, branch_data,1,cap,1,1)

components = False
if components:
    compgraph = nx.Graph()
    net_demand_labels = dict()
    i = 1
    for component in nx.connected_components(graph):
        print(len(component))
        demand = sum(graph.nodes[node]['bus_pd'] for node in component)
        generation = sum(graph.nodes[node]['gen_Pmax'] for node in component)
        net_demand_labels[i] = demand - generation
        compgraph.add_node(i,net_demand = demand - generation)
        i+=1

    for item in net_demand_labels:
        print(item,net_demand_labels[item])
    print(sum(net_demand_labels.values()))

    nx.draw(compgraph,  with_labels=True, labels=net_demand_labels)
    plt.show()
else:
    plt.figure(figsize=(200,200), dpi=60)
    nx.draw(graph, with_labels=True, pos=nx.kamada_kawai_layout(graph))
    plt.savefig('immense_graph.pdf')
    plt.show()