# -*- coding: utf-8 -*-
"""
Created on Fri May 28 12:09:38 2021

@author: Kyle Skolfield
"""


import gurobipy as gp
import networkx as nx
import numpy as np
import mpinput as mp
import itertools
import time
import pandas as pd
import time
#import cProfile
import sys
from itertools import islice
#from itertools import combinations
import random
import math
#import nicify_tree_decomp as ntd
#import os


##########################################
########### VALID INEQUALITIES ###########
##########################################

def cycle_basis_VI(model, where):
    '''if where == gp.GRB.Callback.MIP:
        run_time = model.cbGet(gp.GRB.Callback.RUNTIME)
        mip_gap = model.params.MIPGap
        if run_time > time_of_change and mip_gap != new_gap:
            model._changeParam = True
            model.terminate()'''

    if where == gp.GRB.Callback.MIPSOL:
        if len(cycle_paths) <= 0:
            pass
        path = random.choice(cycle_paths)
        cycle_paths.remove(path)
        path1 = list(nx.utils.pairwise(path[0]))
        path2 = list(nx.utils.pairwise(path[1]))
        mp_length = graph.edge_subgraph(path1).size(weight='branch_CR')
        start = path[0][0]
        end = path[-1][1]

        sp_e = path2
        sp_length = graph.edge_subgraph(sp_e).size(weight='branch_CR')

        if sp_length > mp_length:
            sp_e, path1 = path1, sp_e
            sp_length, mp_length = mp_length, sp_length

        for i in range(len(path1)):
            if path1[i][0] > path1[i][1]:
                path1[i] = tuple(reversed(path1[i]))
        for i in range(len(sp_e)):
            if sp_e[i][0] > sp_e[i][1]:
                sp_e[i] = tuple(reversed(sp_e[i]))

        edge_status = nx.get_edge_attributes(graph, 'branch_status')
        model.cbLazy(model._bus_angle[start] - model._bus_angle[end] <= sp_length +
                     (mp_length - sp_length) * (len(sp_e) - gp.quicksum([model._expansion[edge] for edge in sp_e if edge_status[edge] == 0])))
        model.cbLazy(model._bus_angle[end] - model._bus_angle[start] <= sp_length +
                     (mp_length - sp_length) * (len(sp_e) - gp.quicksum([model._expansion[edge] for edge in sp_e if edge_status[edge] == 0])))


                             
##############################################
########### END VALID INEQUALITIES ###########
##############################################

if len(sys.argv) > 1:
    bus_fail = float(sys.argv[1])
    line_fail = float(sys.argv[2])
else:
    bus_fail = .05
    line_fail = .05

#############################################
########### EXECUTION BEGINS ################
#############################################


#random.seed(pySEED)

filename = "pglib-opf-master/az_2021_case892_ots.m"
filename = "pglib-opf-master/pglib_opf_case2000_goc.m"

cap = 1
demand = 1
gen = 1
#check scale here
gen_cost = .001


bus_data = np.array([])
gen_data = np.array([])
branch_data = np.array([])
bus_data,gen_data,branch_data = mp.load_data(filename)

graph = nx.Graph()
mp.encode_graph(graph, bus_data, gen_data, branch_data,demand,cap,gen,gen_cost)

cycle_paths = []
cycleBasis = nx.cycle_basis(graph)
for cycle in cycleBasis:
    for node in cycle[1:-1]:
        path1 = cycle[:cycle.index(node) + 1]
        path2 = cycle[cycle.index(node) + 1:]
        path2.append(cycle[0])
        cycle_paths.append((path1, path2))

master_mod = gp.Model()
master_mod.modelSense = gp.GRB.MINIMIZE
master_mod.Params.LogFile = 'ots_log_1.txt'
#master_mod.Params.OutputFlag = 0
master_mod.Params.lazyConstraints = 1
#master_mod.Params.MIPFocus = 1
#master_mod.Params.PreCrush = 1



#scenario s will have s[(i,j)] for line status, s[i] = (bus,gen) for bus/gen status
#1 = instact

scenarios = dict()
bus_status = dict()
line_status = dict()

for bus in graph.nodes:
    x = random.random()
    y = random.random()
    if x >= 1 - bus_fail and y>= 1 - bus_fail:
        bus_status[bus] = (0,0)
    elif y >= 1 - bus_fail and x < 1 - bus_fail:
        bus_status[bus] = (1,0)
    elif x >=1 - bus_fail and y < 1 - bus_fail:
        bus_status[bus] = (0,1)
    else:
        bus_status[bus] = (1,1)

for edge in graph.edges:
    x = random.random()
    if x>= 1 - line_fail:
        line_status[edge] = 0
    else:
        line_status[edge] = 1

scenarios[1] = dict()

for node in graph.nodes:
    scenarios[1][node] = bus_status[node]

for edge in graph.edges:
    scenarios[1][edge] = line_status[edge]

for i in range(2,101):
    scenarios[i] = dict()
    temp_nodes = list(bus_status.values())
    temp_edges = list(line_status.values())
    random.shuffle(temp_nodes)
    random.shuffle(temp_edges)
    new_nodes = dict(zip(bus_status,temp_nodes))
    new_edges = dict(zip(line_status,temp_edges))
    for node in graph.nodes:
        scenarios[i][node] = new_nodes[node]

    for edge in graph.edges:
        scenarios[i][edge] = new_edges[edge]

#for i in range(1,2):
#    scenarios[i] = dict()


line_scenarios = list(itertools.product(graph.edges, scenarios))
node_scenarios = list(itertools.product(graph.nodes, scenarios))



#Gather line status and cost properties for full graph
M = 2*.6*max({key: 1/value for  (key,value) in nx.get_edge_attributes(graph,'branch_b').items()}.values())
edge_b = nx.get_edge_attributes(graph, 'branch_b')
Pi = 500
hard_bus_cost = 1
hard_gen_cost = 10
hard_line_cost = 5
investments=[]

#Add binary decision variables

master_mod._switch = master_mod.addVars(line_scenarios, vtype = gp.GRB.BINARY, name = 'switch_status')
master_mod._bus_invest = master_mod.addVars(graph.nodes, vtype = gp.GRB.BINARY, name = 'bus_investment', obj=hard_bus_cost)
master_mod._gen_invest = master_mod.addVars(graph.nodes, vtype = gp.GRB.BINARY, name = 'gen_investment', obj=hard_gen_cost)
master_mod._line_invest = master_mod.addVars(graph.edges, vtype = gp.GRB.BINARY, name = 'line_investment', obj=hard_line_cost)



#TODO: update budget constraint


#Budget constraint goes here
master_mod._budget = master_mod.addConstr(hard_bus_cost * gp.quicksum(master_mod._bus_invest) +
                                          hard_gen_cost * gp.quicksum(master_mod._gen_invest) +
                                          hard_line_cost * gp.quicksum(master_mod._line_invest) <= Pi)


#Add standard variables

######## Flow Variables #######
master_mod._corr_flow = master_mod.addVars(line_scenarios, name='corr_flow', ub = gp.GRB.INFINITY, lb = -gp.GRB.INFINITY)


######## Generation Variables ########
master_mod._gen = master_mod.addVars(node_scenarios, name = 'gen', ub = nx.get_node_attributes(graph, 'gen_Pmax'))
                       #lb= nx.get_node_attributes(graph, 'gen_Pmin'), obj = nx.get_node_attributes(graph, 'gen_cost'))

#master_mod.addConstrs(master_mod._gen[i,s] <=nx.get_node_attributes(graph,'gen_Pmax')[i] * (scenarios[s][i][1] + master_mod._gen_invest[i])  for (i,s) in node_scenarios)

######## Angle Variables ########
master_mod._bus_angle = master_mod.addVars(node_scenarios, name = 'bus_angle', ub = 30, lb = -30)


######## Load Shed Variables #######
shed_cost = max(nx.get_node_attributes(graph,'gen_cost').values())
shed_cost = 1
shed_cap = nx.get_node_attributes(graph, 'bus_pd')
for node in graph.nodes:
    shed_cap[node] *= 1
master_mod._shed = master_mod.addVars(node_scenarios, name = "load_shed", lb = 0, ub = shed_cap,  obj=shed_cost)


master_mod.addConstrs(( - master_mod._corr_flow[(i, j), s] + edge_b[i,j] * (master_mod._bus_angle[i,s] - master_mod._bus_angle[j,s]) +
                       M * (1 - master_mod._switch[(i, j), s]) >= 0 for ((i, j), s) in line_scenarios))
master_mod.addConstrs((-master_mod._corr_flow[(i, j), s] + edge_b[i,j] * (master_mod._bus_angle[i,s] - master_mod._bus_angle[j,s])
                       - M * (1 - master_mod._switch[(i, j), s]) <= 0 for ((i, j), s) in line_scenarios))

master_mod.addConstrs((master_mod._corr_flow[(i, j), s] <= graph.edges[i, j]['branch_cap'] * master_mod._switch[(i, j), s] for ((i, j), s) in line_scenarios))
master_mod.addConstrs((master_mod._corr_flow[(i, j), s] >= -graph.edges[i, j]['branch_cap'] * master_mod._switch[(i, j), s] for ((i, j), s) in line_scenarios))

master_mod.addConstrs(gp.quicksum([master_mod._corr_flow[(j, i), s] for j in graph.neighbors(i) if j < i]) -
                      gp.quicksum([master_mod._corr_flow[(i, j), s] for j in graph.neighbors(i) if j > i]) +
                      master_mod._gen[i,s] + master_mod._shed[i, s] == nx.get_node_attributes(graph, 'bus_pd')[i] for (i, s) in node_scenarios)

master_mod.addConstrs(master_mod._switch[(i, j), s] <= scenarios[s][(i,j)] + master_mod._line_invest[i, j] for ((i, j), s) in line_scenarios)
master_mod.addConstrs(master_mod._switch[(i, j), s] <= scenarios[s][i][0] + master_mod._bus_invest[i] for ((i, j), s) in line_scenarios if j in graph.neighbors(i))
master_mod.addConstrs(master_mod._switch[(i, j), s] <= scenarios[s][j][0] + master_mod._bus_invest[j] for ((i, j), s) in line_scenarios if i in graph.neighbors(j))
master_mod.addConstrs(master_mod._gen[i, s] <= nx.get_node_attributes(graph,'gen_Pmax')[i] * (scenarios[s][i][1] + master_mod._gen_invest[i]) for (i,s) in node_scenarios)
master_mod.addConstrs(master_mod._shed[i, s] >= nx.get_node_attributes(graph, 'bus_pd')[i] * (1 - scenarios[s][i][0] - master_mod._bus_invest[i]) for (i, s) in node_scenarios)
###############################################
############# End Master Problem ##############
###############################################

###############################################
############## Begin Iteration ################
###############################################


master_mod.Params.MIPGap = .001
time_of_change = 600
new_gap = .01
master_mod._changeParam = False

master_mod.optimize()
while master_mod._changeParam and new_gap <= .07:
    master_mod.params.MIPGap = new_gap
    #time_of_change += 600
    new_gap += .01

    master_mod.optimize()


print(master_mod.status)
if master_mod.status == 3:
    master_mod.computeIIS()
    master_mod.write("master_IIS.ilp")
master_mod.write("master_mod.lp")
master_mod.write("ots_200_scenario.sol")

print("Solution to Master Problem: " + str(master_mod.objVal))
print()

b_count = 0
g_count = 0
l_count = 0
r_cost = 0
e_cost = 0
for var in master_mod.getVars():
    if 'bus_invest' in var.VarName:
        if abs(var.x) >= .0001:
            b_count += 1
    elif 'gen_invest' in var.VarName:
        if abs(var.x) >= .0001:
            g_count += 1
    elif 'line_invest' in var.VarName:
        if abs(var.x) >= .0001:
            l_count += 1
            #print(var.VarName, var.x)
investments.append((b_count,g_count,l_count,master_mod.objVal))



output_file = 'output_results_bus'+ str(bus_fail)[1:] + '_line' + str(line_fail)[1:] + '.txt'


with open(output_file,'w') as out:
    shed = 0
    for var in master_mod.getVars():
        if var.Varname[0:4] == 'load':
            shed += var.x

    out.write('shed (MW)' + '\t' + str(shed) + '\n')
    out.write('shed cost' + '\t' + str(shed_cost) + '\n')

    for iter in investments:
        out.write('investment' + '\t' + str(iter) + '\n')

    for var in master_mod.getVars():
        if var.VarName[0] == 'r' or var.VarName[0] == 'e':
            out.write(var.varName + '\t' + str(var.x) + '\n')

print("total load shed (MW): ", shed)
print("investments (bus,gen,line): ", investments)
print()
print()
print('node, demand, shed')
for node in graph.nodes:
    names = 'load_shed[' +str(node) + ',1]'
    print(node, graph.nodes[node]['bus_pd'], master_mod.getVarByName(names).x)