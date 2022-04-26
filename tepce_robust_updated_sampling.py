# -*- coding: utf-8 -*-
"""
Created on Fri May 28 12:09:38 2021

@author: Kyle Skolfield
"""


import gurobipy as gp
import networkx as nx
import numpy as np
import mpinput as mp
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


########################################
########### HELPER FUNCTIONS ###########
########################################


def path_lengths(graph,path_set):
    lengths = []
    for path in path_set:
        length = 0
        for edge in path:
            length += graph.edges[edge]['branch_CR']
        lengths.append(length)
    return lengths

def k_shortest_paths(G, source, target, k, weight=None):
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))

#returns edge list of a decently long path between start node and end node
#may not actually reach end node
#what to do if it doesn't reach?
def path_expander(graph,start_node,end_node):
    used_v = []
    used_v.append(start_node)
    curr_node = start_node
    main_subgraph = graph.copy()
    while len(main_subgraph.nodes) > 1:
        #print(curr_node, list(main_subgraph.nodes))
        curr_weight = 0
        neighbors = nx.neighbors(main_subgraph,curr_node)
        main_subgraph.remove_node(curr_node)
        for adj_node in neighbors:
            reachable_nodes = nx.dfs_preorder_nodes(main_subgraph,adj_node)
            temp_subgraph = nx.subgraph(main_subgraph,reachable_nodes)
            temp_weight = temp_subgraph.size(weight = 'branch_CR')
            if  temp_weight >= curr_weight:
                curr_weight = temp_weight
                new_node = adj_node
        used_v.append(new_node)
        if curr_node == new_node or curr_node == end_node:
            break
        curr_node = new_node
    return list(nx.utils.pairwise(used_v))

def induced_path(start_node,graph):
    temp_graph = graph.copy()
    neighbors = list(temp_graph.neighbors(start_node))
    path_nodes = [start_node]
    while len(neighbors) > 0:
        next_node = random.choice(neighbors)
        #print('traveling to node', next_node)
        path_nodes.append(next_node)
        #print('path so far', path_nodes)
        neighbors.remove(next_node)
        for node in neighbors:
            temp_graph.remove_node(node)
            #print('removing node', node)
            neighbors.remove(node)
        neighbors = list(set(temp_graph.neighbors(next_node)) - set(path_nodes))
    path_edges = list(nx.utils.pairwise(path_nodes))
    temp_graph = temp_graph.edge_subgraph(path_edges).copy()
        
    return temp_graph

def long_induced_path(graph, repeats):
    best_length = 0
    #can specificy subset of nodes
    subset = graph.nodes
    for node in subset:
        #temp_best = 0
        #print(node)
        for i in range(repeats):
            path_graph = induced_path(node,graph)
            path_length = path_graph.size(weight='branch_CR')
            if path_length > best_length:
                best_length = path_length
    return best_length

def ampacity(temp):
    q_conv = 8550*(200-temp)*(0.5*21*.001)**0.448
    q_rad = 0.6*0.000000057*math.pi*21*((200+273)**4 - (temp + 273)**4)
    q_solar = 0.5*1200*21
    r = 0.05*(1 + 0.00403*(200-20))
    return math.sqrt((q_conv + q_rad + q_solar)/r)/2100

###################################
########### END HELPERS ###########
###################################



########################################
########### BASE DCOPF MODEL ###########
########################################

#For full DCOPF, use flow capacity constraints from OTS and force all switch variables to be constant 1

#Load flow equations
def load_cons_flow_eq(mod,graph):
    M = 2*.6*max({key: 1/value for  (key,value) in nx.get_edge_attributes(graph,'branch_x').items()}.values())
    #M = 2*6.8
    edge_x = nx.get_edge_attributes(graph, 'branch_x')
    mod.addConstrs((-1/edge_x[i,j] * (mod._bus_angle[i] - mod._bus_angle[j] ) - mod._corr_flow[i,j] + (1 - mod._switch[i,j])*M >= 0 for (i,j) in mod._corr_flow),
                   name = "F_eq_pos")
    mod.addConstrs((-1/edge_x[i,j] * (mod._bus_angle[i] - mod._bus_angle[j] ) - mod._corr_flow[i,j] - (1 - mod._switch[i,j])*M <= 0 for (i,j) in mod._corr_flow),
                   name = "F_eq_neg")

#Load bus angle difference constraints
def load_cons_bus_ang_diff(mod,graph,bus_ang):
    mod.addConstrs((mod._bus_angle[i] - mod._bus_angle[j] <= bus_ang for (i,j) in mod._corr_flow))
    mod.addConstrs((mod._bus_angle[j] - mod._bus_angle[i] <= bus_ang for (i,j) in mod._corr_flow))
  

#Load flow balance constraints
def load_cons_balance(mod,graph):
    for i in graph.nodes:
        mod.addConstr(gp.quicksum([mod._corr_flow[j,i] for j in graph.neighbors(i) if j < i]) - 
                      gp.quicksum([mod._corr_flow[i,j] for j in graph.neighbors(i) if j > i]) + 
                      (mod._gen.get(i) or 0) == nx.get_node_attributes(graph, 'bus_pd')[i])

#######################################
########### END DCOPF MODEL ###########
#######################################




#######################################
########### OTS CONSTRAINTS ###########
#######################################

#Load flow capacity constraints
def load_cons_OTS_flow_cap(mod,graph):
    edge_caps = nx.get_edge_attributes(graph,'branch_cap')
    mod.addConstrs((mod._corr_flow[i,j] <= edge_caps[i,j] * mod._switch[i,j] for (i,j) in mod._corr_flow), name="F_cap_pos")
    mod.addConstrs((mod._corr_flow[i,j] >= -edge_caps[i,j] * mod._switch[i,j] for (i,j) in mod._corr_flow), name="F_cap_neg")

#Load maximum number of switched off lines constraint
def load_cons_switch_cap(mod,graph,cap):
    mod.addConstr(len(mod._corr_flow) - gp.quicksum(mod._switch) <=cap, name='switch_cap')
     
###########################################
########### END OTS CONSTRAINTS ###########
###########################################



#######################################
########### TEP CONSTRAINTS ###########
#######################################

def load_cons_flow_eq_TEP(mod,graph):
    M = 2*.6*max({key: 1/value for  (key,value) in nx.get_edge_attributes(graph,'branch_x').items()}.values())
    #M = 2*6.8
    edge_x = nx.get_edge_attributes(graph, 'branch_x')
    mod.addConstrs((-1/edge_x[i,j] * (mod._bus_angle[i] - mod._bus_angle[j] ) - mod._corr_flow[i,j] + 
                    (1 - mod._new_line[i,j])*M >= 0 for (i,j) in mod._corr_flow if edge_status[i,j] == 0),
                   name = "F_eq_pos")
    mod.addConstrs((-1/edge_x[i,j] * (mod._bus_angle[i] - mod._bus_angle[j] ) - mod._corr_flow[i,j] - 
                    (1 - mod._new_line[i,j])*M <= 0 for (i,j) in mod._corr_flow if edge_status[i,j] == 0),
                   name = "F_eq_neg")
    mod.addConstrs((-1/edge_x[i,j] * (mod._bus_angle[i] - mod._bus_angle[j] ) ==  mod._corr_flow[i,j] for (i,j) in mod._corr_flow if edge_status[i,j] == 1),
                   name = "F_eq")


def load_TEP_cons_flow_cap(mod,graph):
    edge_caps = nx.get_edge_attributes(graph,'branch_cap')
    edge_status = nx.get_edge_attributes(graph,'branch_status')
    mod.addConstrs((mod._corr_flow[i,j] <= edge_caps[i,j] * mod._new_line[i,j] for (i,j) in mod._corr_flow if edge_status[i,j] == 0), name="F_cap_pos")
    mod.addConstrs((mod._corr_flow[i,j] >= -edge_caps[i,j] * mod._new_line[i,j] for (i,j) in mod._corr_flow if edge_status[i,j] == 0), name="F_cap_neg")
    mod.addConstrs((mod._corr_flow[i,j] <= edge_caps[i,j] + edge_caps[i,j] * mod._cap_exp[i,j]  for (i,j) in mod._corr_flow if edge_status[i,j] == 1), name="F_cap_pos")
    mod.addConstrs((mod._corr_flow[i,j] >= -edge_caps[i,j] - edge_caps[i,j] * mod._cap_exp[i,j] for (i,j) in mod._corr_flow if edge_status[i,j] == 1), name="F_cap_neg")
    
    
def load_cons_flow_eq_TEP_hyb(mod,graph):
    edge_x = nx.get_edge_attributes(graph, 'branch_x')
    mod.addConstrs((-1/edge_x[i,j] * (mod._bus_angle[i] - mod._bus_angle[j] ) ==  mod._corr_flow[i,j] for (i,j) in mod._corr_flow if edge_status[i,j] == 1),
                   name = "F_eq")

###########################################
########### END TEP CONSTRAINTS ###########
###########################################



def fused_graph_pos(graph, model_list, lin_mod, hyb_mod, trans_mod):
    pos_flow = graph.copy()
    
    for edge in pos_flow.edges:
        if model_list == ['lin']:
            if lin_mod._corr_flow[edge].x <= 0:
                pos_flow.remove_edge(edge[0],edge[1])
        elif model_list == ['hyb']:
            if hyb_mod._corr_flow[edge].x <= 0:
                pos_flow.remove_edge(edge[0],edge[1])
        elif model_list == ['trans']:
            if trans_mod._corr_flow[edge].x <= 0:
                pos_flow.remove_edge(edge[0],edge[1])
        elif 'lin' in model_list and 'hyb' in model_list:
            if lin_mod._corr_flow[edge].x <= 0 or hyb_mod._corr_flow[edge].x <=0:
                pos_flow.remove_edge(edge[0],edge[1])
        elif 'lin' in model_list and 'trans' in model_list:
            if lin_mod._corr_flow[edge].x <= 0 or trans_mod._corr_flow[edge].x <=0:
                pos_flow.remove_edge(edge[0],edge[1])
        elif 'trans' in model_list and 'hyb' in model_list:
            if trans_mod._corr_flow[edge].x <= 0 or hyb_mod._corr_flow[edge].x <=0:
                pos_flow.remove_edge(edge[0],edge[1])
        elif 'lin' in model_list and 'hyb' in model_list and 'trans' in model_list:
            if lin_mod._corr_flow[edge].x <= 0 or hyb_mod._corr_flow[edge].x <=0 or trans_mod._corr_flow[edge].x <= 0:
                pos_flow.remove_edge(edge[0],edge[1]) 
    
    return pos_flow
        

def fused_graph_neg(graph, model_list, lin_mod, hyb_mod, trans_mod):
    neg_flow = graph.copy()
    
    for edge in neg_flow.edges:
        if model_list == ['lin']:
            if lin_mod._corr_flow[edge].x >= 0:
                neg_flow.remove_edge(edge[0],edge[1])
        elif model_list == ['hyb']:
            if hyb_mod._corr_flow[edge].x >= 0:
                neg_flow.remove_edge(edge[0],edge[1])
        elif model_list == ['trans']:
            if trans_mod._corr_flow[edge].x >= 0:
                neg_flow.remove_edge(edge[0],edge[1])
        elif 'lin' in model_list and 'hyb' in model_list:
            if lin_mod._corr_flow[edge].x >= 0 or hyb_mod._corr_flow[edge].x >=0:
                neg_flow.remove_edge(edge[0],edge[1])
        elif 'lin' in model_list and 'trans' in model_list:
            if lin_mod._corr_flow[edge].x >= 0 or trans_mod._corr_flow[edge].x >=0:
                neg_flow.remove_edge(edge[0],edge[1])
        elif 'trans' in model_list and 'hyb' in model_list:
            if trans_mod._corr_flow[edge].x >= 0 or hyb_mod._corr_flow[edge].x >=0:
                neg_flow.remove_edge(edge[0],edge[1])
        elif 'lin' in model_list and 'hyb' in model_list and 'trans' in model_list:
            if lin_mod._corr_flow[edge].x >= 0 or hyb_mod._corr_flow[edge].x >=0 or trans_mod._corr_flow[edge].x >= 0:
                neg_flow.remove_edge(edge[0],edge[1]) 
    
    return neg_flow



##########################################
########### VALID INEQUALITIES ###########
##########################################

def cycle_basis_VI(model, where):
    if where == gp.GRB.Callback.MIP:
        run_time = model.cbGet(gp.GRB.Callback.RUNTIME)
        mip_gap = model.params.MIPGap
        if run_time > time_of_change and mip_gap != new_gap:
            model._changeParam = True
            model.terminate()
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


def create_subproblem(mast_mod, graph,k,shed_cost):
        
        if k >= 1:
            capstr ='branch_cap' + str(k)
        else:
            capstr = 'branch_cap'
    
        mod_lin = gp.Model()
        #mod_lin.Params.OutputFlag = 0
        #mod_lin.Params.PreSolve = 0
        
        ######## Flow Variables #######
        #Dictionary comprehension for negatives
        mod_lin._corr_flow = mod_lin.addVars(graph.edges, name='corr_flow', 
                                     ub = gp.GRB.INFINITY, 
                                     lb = -gp.GRB.INFINITY)
        
    
        ######## Generation Variables ########
        mod_lin._gen = mod_lin.addVars(graph.nodes, name = 'gen', obj=nx.get_node_attributes(graph, 'gen_cost'))
        #shed_cost = max(nx.get_node_attributes(graph,'gen_cost').values())
        #shed_cost = 100000
        mod_lin._shed = mod_lin.addVars(graph.nodes, name = 'load_shed', lb= 0, obj = shed_cost)
        
        ######## Angle Variables ########
        mod_lin._bus_angle = mod_lin.addVars(graph.nodes, name = 'bus_angle', ub = gp.GRB.INFINITY, lb = -gp.GRB.INFINITY)
        
    
        
        ######## Load All Constraints ########
        mod_lin.addConstrs(gp.quicksum([mod_lin._corr_flow[j,i] for j in graph.neighbors(i) if j < i]) - 
                           gp.quicksum([mod_lin._corr_flow[i,j] for j in graph.neighbors(i) if j > i]) + 
                           (mod_lin._gen.get(i) or 0) + mod_lin._shed[i] == nx.get_node_attributes(graph, 'bus_pd')[i] for i in graph.nodes)
        
        
        #mod_lin.addConstrs((mod_lin._bus_angle[i] - mod_lin._bus_angle[j] <= 30 for (i,j) in mod_lin._corr_flow))
        #mod_lin.addConstrs((mod_lin._bus_angle[j] - mod_lin._bus_angle[i] <= 30 for (i,j) in mod_lin._corr_flow))
        
        mod_lin.addConstrs(mod_lin._bus_angle[i] <= 30 for i in graph.nodes)
        mod_lin.addConstrs(mod_lin._bus_angle[i] >= -30 for i in graph.nodes)
        
        
        edge_b = nx.get_edge_attributes(graph, 'branch_b')
        edge_caps = nx.get_edge_attributes(graph,capstr)
        
        edge_status = nx.get_edge_attributes(graph,'branch_status')
        
        mod_lin.addConstrs(((mod_lin._bus_angle[i] - mod_lin._bus_angle[j] ) + (-1/edge_b[i,j]) * mod_lin._corr_flow[i,j] +
                        (1 - mast_mod._expansion[i,j].x)*M >= 0 for (i,j) in mod_lin._corr_flow if edge_status[i,j] == 0),
                       name = "F_eq_pos")
        mod_lin.addConstrs(((mod_lin._bus_angle[i] - mod_lin._bus_angle[j] ) + (-1/edge_b[i,j]) * mod_lin._corr_flow[i,j] -
                        (1 - mast_mod._expansion[i,j].x)*M <= 0 for (i,j) in mod_lin._corr_flow if edge_status[i,j] == 0),
                       name = "F_eq_neg")
        mod_lin.addConstrs(((mod_lin._bus_angle[i] - mod_lin._bus_angle[j] ) == (-1/edge_b[i,j]) * mod_lin._corr_flow[i,j] for (i,j) in mod_lin._corr_flow if edge_status[i,j] == 1),
                       name = "F_eq")
        
        mod_lin.addConstrs(mod_lin._gen[i] <= graph.nodes[i]['gen_Pmax'] for i in graph.nodes)
        mod_lin.addConstrs(mod_lin._shed[i] <= graph.nodes[i]['bus_pd'] for i in graph.nodes)
        
        
        
        mod_lin.addConstrs((mod_lin._corr_flow[i,j] <= edge_caps[i,j] * mast_mod._expansion[i,j].x for (i,j) in mod_lin._corr_flow if edge_status[i,j] == 0), name="F_cap_pos")
        mod_lin.addConstrs((mod_lin._corr_flow[i,j] >= -edge_caps[i,j] * mast_mod._expansion[i,j].x for (i,j) in mod_lin._corr_flow if edge_status[i,j] == 0), name="F_cap_neg")
        mod_lin.addConstrs((mod_lin._corr_flow[i,j] <= edge_caps[i,j] + edge_caps[i,j] * mast_mod._reconductor[i,j].x  for (i,j) in mod_lin._corr_flow if edge_status[i,j] == 1), name="F_cap_pos")
        mod_lin.addConstrs((mod_lin._corr_flow[i,j] >= -edge_caps[i,j] - edge_caps[i,j] * mast_mod._reconductor[i,j].x for (i,j) in mod_lin._corr_flow if edge_status[i,j] == 1), name="F_cap_neg")
    
        return mod_lin



##############################################
############ INPUT SETTINGS ##################
##############################################




#############################################
########### EXECUTION BEGINS ################
#############################################

filenames = ["pglib-opf-master/az_2022_case892.m"]
#filenames = ["pglib-opf-master/pglib_opf_case500_GOC.m"]

#LP_Length = 3015
cap = 2
demand = 1
gen = 1
#check scale here
gen_cost = .0001

for filename in filenames:
    
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

    ###############################################
    ########### Declare Master Problem ############
    ###############################################

    master_mod = gp.Model()
    master_mod.modelSense = gp.GRB.MINIMIZE
    master_mod.Params.LogFile = 'master_mod_test_2.txt'
    #default .001
    master_mod.Params.MIPGap = .01
    #master_mod.Params.OutputFlag = 0
    master_mod.Params.lazyConstraints = 1
    #master_mod.Params.MIPFocus = 1
    #master_mod.Params.PreCrush = 1
    
    #Gather line status and cost properties for full graph
    #M = 2*.6*max({key: value for  (key,value) in nx.get_edge_attributes(graph,'branch_b').items()}.values())
    M = 2 * .6 * max({key: 1/value for (key, value) in nx.get_edge_attributes(graph, 'branch_b').items()}.values())
    #M = 1000000
    edge_status = nx.get_edge_attributes(graph,'branch_status')
    edge_b = nx.get_edge_attributes(graph, 'branch_b')
    expand_lines = [(i,j) for (i,j) in graph.edges if edge_status[i,j] ==0]
    recond_lines = [(i,j) for (i,j) in graph.edges if edge_status[i,j] ==1]
    expand_cost = nx.get_edge_attributes(graph,'branch_exp_cost')
    recond_cost = nx.get_edge_attributes(graph,'branch_cand_cost')
    
    for cost in expand_cost:
        expand_cost[cost] *= .1
        recond_cost[cost] *= 1
    
    #Add binary decision variables
    #eq 3.56
    master_mod._expansion = master_mod.addVars(expand_lines, vtype = gp.GRB.BINARY, name = 'expansion_status', obj = expand_cost)
    master_mod._reconductor = master_mod.addVars(recond_lines, vtype = gp.GRB.BINARY, name = 'reconductor_status', obj = recond_cost)
    
    
    #Add lb based on subproblems
    #eq 3.56
    master_mod._gamma = master_mod.addVar(name = 'gamma', obj = 1)
    
    #Budget constraint goes here
    #eq 3.57
    Pi = 50000000
    master_mod._budget = master_mod.addConstr(gp.quicksum([expand_cost[i,j]*master_mod._expansion[i,j] for (i,j) in expand_lines]) +
                                              gp.quicksum([recond_cost[i,j]*master_mod._reconductor[i,j] for (i,j) in recond_lines]) <= Pi)
    
    
    #Add standard variables
    
    
    ######## Flow Variables #######
    master_mod._corr_flow = master_mod.addVars(graph.edges, name='corr_flow', ub = gp.GRB.INFINITY, lb = -gp.GRB.INFINITY)
    

    ######## Generation Variables ########
    #master_mod._gen = master_mod.addVars(nx.get_node_attributes(graph, 'gen_cost'), name = 'gen', ub = nx.get_node_attributes(graph, 'gen_Pmax'),
                           #lb= nx.get_node_attributes(graph, 'gen_Pmin'))
    #eq 3.30
    master_mod._gen = master_mod.addVars(graph.nodes, name='gen', ub=nx.get_node_attributes(graph, 'gen_Pmax'), lb=0)
    
    ######## Angle Variables ########
    #eq 3.31,3.32
    master_mod._bus_angle = master_mod.addVars(graph.nodes, name = 'bus_angle', ub = 30, lb = -30)
    
    
    ######## Load Shed Variables #######
    shed_cost = max(nx.get_node_attributes(graph,'gen_cost').values())
    #shed_cost = 1000000000000000
    shed_cap = nx.get_node_attributes(graph, 'bus_pd')
    for node in graph.nodes:
        shed_cap[node] *= 1
    #eq 3.33
    master_mod._shed = master_mod.addVars(graph.nodes, name = "load_shed", lb = 0, ub = shed_cap, obj = 10000000)
    #master_mod.addConstr(gp.quicksum(master_mod._shed) <= 0.1*sum(nx.get_node_attributes(graph, 'bus_pd').values()))
    #master_mod.addConstr(master_mod._gamma == gp.quicksum([shed_cost*master_mod._shed[i] for i in graph.nodes]) +
     #                    gp.quicksum([graph.nodes[i]['gen_cost']*master_mod._gen[i] for i in graph.nodes]))
    
    
    ###############################################
    ############# End Master Problem ##############
    ###############################################
    
    
    
    ###############################################
    ############# Declare Sub Problem #############
    ###############################################
    
    
    sub_mod = gp.Model()
    sub_mod.modelSense = gp.GRB.MAXIMIZE
    sub_mod.Params.LogFile = 'sub_mod_test_iter_1.txt'
    
    
    #Decare variabes.  Objective coefficiencts to be set later for all but lambda_dual
    #eq 3.40
    sub_mod._lambda = sub_mod.addVars(graph.nodes, lb = - gp.GRB.INFINITY, ub = gp.GRB.INFINITY, name = 'lambda_dual')
    #eq 3.41
    sub_mod._chi_hat = sub_mod.addVars(recond_lines, lb = - gp.GRB.INFINITY, ub = 0, name = "chi_hat_dual")
    #eq 3.42
    sub_mod._chi_check = sub_mod.addVars(recond_lines, lb = 0, ub = gp.GRB.INFINITY, name = "chi_check_dual")
    # eq 3.43
    sub_mod._phi_hat = sub_mod.addVars(expand_lines, lb = - gp.GRB.INFINITY, ub = 0, name = "phi_hat_dual")
    # eq 3.44
    sub_mod._phi_check = sub_mod.addVars(expand_lines, lb = 0 , ub = gp.GRB.INFINITY, name = "phi_check_dual")
    # eq 3.45
    sub_mod._xi = sub_mod.addVars(recond_lines, lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY, name = "xi_dual")
    # eq 3.46
    sub_mod._xi_hat = sub_mod.addVars(expand_lines, lb = -gp.GRB.INFINITY, ub = 0, name = "xi_hat_dual")
    # eq 3.47
    sub_mod._xi_check = sub_mod.addVars(expand_lines, lb = 0, ub = gp.GRB.INFINITY, name = "xi_check_dual")
    # eq 3.48
    sub_mod._varphi = sub_mod.addVars(graph.nodes, lb = -gp.GRB.INFINITY, ub = 0, name = "varphi_dual")
    # eq 3.49
    sub_mod._varphi_hat = sub_mod.addVars(graph.nodes, lb = -gp.GRB.INFINITY, ub = 0, name = "varphi_hat_dual")
    # eq 3.50
    sub_mod._varphi_check = sub_mod.addVars(graph.nodes, lb = 0, ub = gp.GRB.INFINITY, name = "varphi_check_dual")
    # eq 3.51
    sub_mod._upsilon = sub_mod.addVars(graph.nodes, lb = -gp.GRB.INFINITY, ub = 0, name = "upsilon_dual")
    
    
    gen_costs = nx.get_node_attributes(graph, 'gen_cost')
    gen_caps = nx.get_node_attributes(graph, 'gen_Pmax')
    
    
    #Adding dual constraints
    #eq 3.35
    sub_mod.addConstrs((sub_mod._lambda[i] + sub_mod._varphi[i] <= gen_costs[i] for i in graph.nodes), name = 'dual_one')
    #eq 3.36
    sub_mod.addConstrs(sub_mod._chi_hat[i,j] + sub_mod._chi_check[i,j] + sub_mod._lambda[j] - sub_mod._lambda[i] 
                       - sub_mod._xi[i,j]*(1 / graph.edges[(i,j)]['branch_b']) == 0 for (i,j) in recond_lines)
    #eq 3.37
    sub_mod.addConstrs(sub_mod._lambda[j] - sub_mod._lambda[i] + sub_mod._phi_hat[i,j] + sub_mod._phi_check[i,j] - 
                       sub_mod._xi_hat[i,j]*(1/graph.edges[(i,j)]['branch_b']) - sub_mod._xi_check[i,j]*(1/graph.edges[(i,j)]['branch_b'])
                       == 0 for (i,j) in expand_lines)
    #eq 3.38
    sub_mod.addConstrs(gp.quicksum([sub_mod._xi[j,i] for j in graph.neighbors(i) if j<i and (j,i) in recond_lines]) +
                       gp.quicksum([sub_mod._xi_hat[j,i] for j in graph.neighbors(i) if j<i and (j,i) in expand_lines]) +
                       gp.quicksum([sub_mod._xi_check[j,i] for j in graph.neighbors(i) if j<i and (j,i) in expand_lines]) -
                       gp.quicksum([sub_mod._xi[i,j] for j in graph.neighbors(i) if j>i and (i,j) in recond_lines]) -
                       gp.quicksum([sub_mod._xi_hat[i,j] for j in graph.neighbors(i) if j>i and (i,j) in expand_lines]) -
                       gp.quicksum([sub_mod._xi_check[i,j] for j in graph.neighbors(i) if j>i and (i,j) in expand_lines]) +
                       sub_mod._varphi_hat[i] + sub_mod._varphi_check[i]
                       == 0 for i in graph.nodes)
    #3.39
    sub_mod.addConstrs(sub_mod._lambda[i] + sub_mod._upsilon[i] <= shed_cost for i in graph.nodes)

    # eq 3.28
    master_mod.addConstrs((master_mod._corr_flow[i, j] + edge_b[i, j] * (master_mod._bus_angle[i] - master_mod._bus_angle[j]) <=
                           M * (1 - master_mod._expansion[i, j]) for (i, j) in expand_lines))
    # eq 3.29
    master_mod.addConstrs((master_mod._corr_flow[i, j] + edge_b[i, j] * (master_mod._bus_angle[i] - master_mod._bus_angle[j]) >=
                           -M * (1 - master_mod._expansion[i, j]) for (i, j) in expand_lines))
    # eq 3.27
    master_mod.addConstrs((master_mod._bus_angle[i] - master_mod._bus_angle[j] == edge_b[i, j] * master_mod._corr_flow[i, j] for (i, j) in recond_lines))
    # eq 3.22
    master_mod.addConstrs(gp.quicksum([master_mod._corr_flow[j, i] for j in graph.neighbors(i) if j < i]) -
                          gp.quicksum([master_mod._corr_flow[i, j] for j in graph.neighbors(i) if j > i]) +
                          master_mod._gen[i] + master_mod._shed[i] == nx.get_node_attributes(graph, 'bus_pd')[i] for i in graph.nodes)
    
    ###############################################
    ############## End Sub Problem ################
    ###############################################

    #sort_flag options -- cluster, month, location, year
    #3020 records per location
    #53 locations
    query_flag = 'summer'
    scale_flag = 1.2

    
    ###############################################
    ############## Begin Iteration ################
    ###############################################

    investments = []
    #sampled_days = [i for i in range(1,129)]
    sheetname = 'Clustering Results Jan_7_22.xlsx'
    temp_data = pd.read_excel(sheetname, header=1, usecols="A:F", engine='openpyxl')

    if query_flag == 'summer':
        grouped_data = temp_data.query('MO == 7 or MO == 8')

    #Sample days from temp_data at uniform -- will abstract process at some point
    #k+=1
    zone_dict = dict()
    sampled_days = dict()
    sampled_length = 100
    for i in range(0,7):
        zone_dict[i] = grouped_data.query('cluster ==' +  str(i))
        sampled_days[i] = zone_dict[i].sample(n=min(sampled_length,len(zone_dict[i])))
    total_days = dict()



    for day in range(0,sampled_length):
        capstr = 'branch_cap' + str(day)
        for (i,j) in graph.edges:
            zone = max(graph.nodes[i]['bus_zone'],graph.nodes[i]['bus_zone'])
            print(zone, len(zone_dict[zone]), day)
            graph.edges[i,j][capstr] = ampacity(sampled_days[zone]['Max'].iloc[day % len(sampled_days[2]['Max'])])*graph.edges[i,j]['branch_cap']

        #Add operating constraints to master problem
        
        #i.e. (3.22) - (3.33) from writeup?

        #eq 3.25
        master_mod.addConstrs((master_mod._corr_flow[i,j] <= graph.edges[i,j][capstr] * master_mod._expansion[i,j] for (i,j) in expand_lines))
        #eq 3.26
        master_mod.addConstrs((master_mod._corr_flow[i,j] >= -graph.edges[i,j][capstr] * master_mod._expansion[i,j] for (i,j) in expand_lines))
        #eq 3.23
        master_mod.addConstrs((master_mod._corr_flow[i,j] <= graph.edges[i,j][capstr] + graph.edges[i,j][capstr] * master_mod._reconductor[i,j]  for (i,j) in recond_lines))
        #eq 3.24
        master_mod.addConstrs((master_mod._corr_flow[i,j] >= -graph.edges[i,j][capstr] - graph.edges[i,j][capstr] * master_mod._reconductor[i,j] for (i,j) in recond_lines))

        
        #coef = random.uniform(.8,1)
        #temporary until stored values
        master_mod.Params.MIPGap = .001
        #Solve Master Problem
        master_mod.update()
        time_of_change = 600
        new_gap = .01
        master_mod._changeParam = False
        #Solve Master Problem
        master_mod.optimize(cycle_basis_VI)
        while master_mod._changeParam and new_gap <= .07:
            master_mod.params.MIPGap = new_gap
            #time_of_change += 600
            new_gap += .01

            master_mod.optimize(cycle_basis_VI)


        print(master_mod.status)

        if master_mod.status == 3:
            master_mod.computeIIS()
            master_mod.write("master_IIS.ilp")
            master_mod.write("master_mod.lp")

        print("Solution to Master Problem: " + str(master_mod.objVal))
        print()
        #eq 3.34
        sub_mod.setObjective(gp.quicksum([nx.get_node_attributes(graph, 'bus_pd')[i]*sub_mod._lambda[i] for i in graph.nodes]) +
                             gp.quicksum([(1 + master_mod._reconductor[i,j].x)*graph.edges[i,j][capstr]*sub_mod._chi_hat[i,j] for (i,j) in recond_lines]) -
                             gp.quicksum([(1 + master_mod._reconductor[i,j].x)*graph.edges[i,j][capstr]*sub_mod._chi_check[i,j] for (i,j) in recond_lines]) +
                             gp.quicksum([graph.edges[i,j][capstr]*master_mod._expansion[i,j].x*sub_mod._phi_hat[i,j] for (i,j) in expand_lines]) -
                             gp.quicksum([graph.edges[i,j][capstr]*master_mod._expansion[i,j].x*sub_mod._phi_check[i,j] for (i,j) in expand_lines]) +
                             gp.quicksum([M*(1-master_mod._expansion[i,j].x)*sub_mod._xi_hat[i,j] for (i,j) in expand_lines]) -
                             gp.quicksum([M*(1-master_mod._expansion[i,j].x)*sub_mod._xi_check[i,j] for (i,j) in expand_lines]) +
                             gp.quicksum([graph.nodes[i]['gen_Pmax']*sub_mod._varphi[i] for i in graph.nodes]) +
                             gp.quicksum([30*sub_mod._varphi_hat[i] for i in graph.nodes]) -
                             gp.quicksum([30*sub_mod._varphi_check[i] for i in graph.nodes]) +
                             gp.quicksum([shed_cap[i] * sub_mod._upsilon[i] for i in graph.nodes]))


        sub_mod.optimize()
        mod_lin = create_subproblem(master_mod, graph, day, shed_cost)
        mod_lin.optimize()
        #sub_mod.write('sub_mod.lp')
        #mod_lin.write('mod_lin.lp')

        print()
        #print("Solution to primal problem:" + str(mod_lin.objVal))

        if sub_mod.status == 2:
            print("Solution to subproblem: " + str(sub_mod.objVal))
            #print("solution to mod_lin: " + str(mod_lin.objVal))
            print()
            #sys.exit()

            #mod_lin.write('primal.lp')
            #sub_mod.write('dual.lp')

            #Add constraint to master problem
            master_mod.addConstr(master_mod._gamma >= sub_mod.objVal)
        #master_mod.addConstr(master_mod._gamma >= mod_lin.objVal)

        #otherwise y and z will never update
        
        random.seed(time.time())
        

        
        r_count = 0
        e_count = 0
        r_cost = 0
        e_cost = 0
        for var in master_mod.getVars():
            if var.VarName[0] == 'r' or var.VarName[0] == 'e':
                if abs(var.x) >= 0.00001:
                    if var.VarName[0] == 'r':
                        r_cost += var.x * var.obj
                        r_count += 1
                    else:
                        e_cost += var.x * var.obj
                        e_count+=1
                    #print(var.VarName, var.x)
        print(r_count,e_count)
        investments.append((r_count,e_count,r_cost,e_cost,master_mod.objVal))



output_file = 'output_results_clustered_sample'+ query_flag + str(time.time())[:6]+ '.txt'


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
    print("investments (recond,expand): ", investments)