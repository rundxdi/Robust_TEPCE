# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 15:13:46 2021

@author: rundx
"""

import mpinput as mp
import networkx as nx
import numpy as np
from networkx import algorithms
import random

filename = 'pglib-opf-master/az_2020_case892.m'

bus_data = np.array([])
gen_data = np.array([])
branch_data = np.array([])
bus_data,gen_data,branch_data = mp.load_data(filename)


cap = 1
graph = nx.Graph()
mp.encode_graph(graph, bus_data, gen_data, branch_data,1,cap,2,1)

print(nx.number_connected_components(graph))

vertices = []

for component in list(nx.connected_components(graph)):
    vertices.append(list(component))
f = open('candidate_lines_az.txt','w')
for vertex in vertices:
    if len(vertex) > 30:
        for i in vertices:
            new_vert_set = i
            temp_vert = random.choice(vertex)
            pair = (temp_vert, random.choice(i))
            f.write(str(pair[0]) + '\t' + str(pair[1]) + '\n')
    else:
        new_vert_set = random.choice(vertices)
        while new_vert_set == vertex:
            new_vert_set = random.choice(vertices)
        pair = (random.choice(vertex),random.choice(new_vert_set))
        f.write(str(pair[0]) + '\t' + str(pair[1]) + '\n')
f.close()
'''

emptygraph = nx.complement(graph)

scale = 1

total_demand = scale*sum(nx.get_node_attributes(graph, 'bus_pd').values())
total_generation = scale*sum(nx.get_node_attributes(graph,'gen_Pmax').values())



for node in emptygraph.nodes:
    emptygraph.nodes[node]['demand'] = scale*(graph.nodes[node]['bus_pd'] - graph.nodes[node]['gen_Pmax'])

for edge in emptygraph.edges:
    emptygraph.edges[edge]['capacity'] = 660000
    emptygraph.edges[edge]['cost'] = 1800000

for edge in graph.edges:
    graph.edges[edge]['cost'] = graph.edges[edge]['branch_cand_cost']
    graph.edges[edge]['capacity'] = 10*graph.edges[edge]['branch_cap']

for node in range(1,893):
    emptygraph.add_edge(node, 1000, cost = 1800000000)

emptygraph.nodes[1000]['demand'] = -sum(nx.get_node_attributes(emptygraph,'demand').values())

join_graph = nx.compose(graph,emptygraph)

join_graph = nx.to_directed(join_graph)



flowcost, flowdict = nx.network_simplex(join_graph, 'demand', 'capacity', 'cost')




graph = nx.to_directed(graph).copy()


for edge in list(graph.edges):
    if edge[0] > edge[1]:
        graph.remove_edge(edge[0],edge[1])
    else:
        graph.edges[edge]['weight'] = -10

graph.add_node('source')
graph.nodes['source']['gen_Pmax'] = 0
graph.add_node('sink')
graph.nodes['sink']['gen_Pmax'] = 0

#graph.add_edge('source','sink')
#graph.edges[('source','sink')]['branch_cap'] = 25999000000000000
#graph.edges[('source','sink')]['weight'] = 1000000000000000000


for node in graph.nodes:
    if node != 'source' and node != 'sink':
        if not graph.nodes[node].get('gen_Pmax') or graph.nodes[node].get('gen_Pmax') == 0:
            graph.add_edge('source',node)
            graph.edges[('source',node)]['branch_cap'] = 1000000
            graph.edges[('source',node)]['weight'] = 12
            graph.nodes[node]['gen_Pmax'] = 0
            graph.add_edge(node,'sink')
            graph.edges[(node,'sink')]['branch_cap'] = 100000
            graph.edges[(node,'sink')]['weight'] = 0
        else:
            graph.nodes[node]['gen_Pmax'] = graph.nodes[node]['gen_Pmax']
            graph.add_edge('source',node)
            graph.edges[('source',node)]['branch_cap'] = graph.nodes[node]['gen_Pmax'] 
            graph.nodes[node]['gen_Pmax'] = 0
            graph.add_edge(node,'sink')
            graph.edges[(node,'sink')]['branch_cap'] = 100000000
            graph.edges[(node,'sink')]['weight'] = 0
        
        
#graph.nodes['sink']['gen_Pmax'] = -sum([graph.nodes[node]['gen_Pmax'] for node in graph.nodes])
#graph.nodes['source']['gen_Pmax'] = sum([graph.nodes[node]['gen_Pmax'] for node in graph.nodes])
graph.nodes['sink']['gen_Pmax'] = 25998.299999999996 
graph.nodes['source']['gen_Pmax'] = -25998.299999999996
  
graph.add_node('super_source')
graph.add_edge('super_source','source')
graph.edges[('super_source','source')]['branch_cap'] = 29579


#25998.299999999996

flows = nx.max_flow_min_cost(graph, 'super_source', 'sink', capacity = 'branch_cap', weight = 'weight')

for node in graph.nodes:
    if node != 'source' and node != 'sink' and node != 'super_source':
        print(flows[node]['sink'])
        #if ('source',node) in graph.edges:
        #    print(flows['source'][node])
        
        
f = open('az_silly_demands.txt','w')
for node in graph.nodes:
    if node != 'source' and node != 'sink' and node != 'super_source':
        f.write(str(flows[node]['sink']) + '\n')
f.close()'''