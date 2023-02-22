import pandas as pd
import os
import numpy as np
import mpinput as mp
import networkx as nx
from pathlib import Path
import geopandas as gp
import matplotlib.pyplot as plt
from shapely.geometry import LineString
import glob
import itertools
import re
import matplotlib

os.makedirs('Robust_TEPCE/TEPCE Results', exist_ok=True)
os.makedirs('TEPCE_Charts/Exp_Rec', exist_ok=True)


def cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j,i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)


def get_bus_pairs():
    target_file = 'TEPCE Results/output_results_clustered_samplesummer165169.txt'
    with open(target_file, "r") as fobj:
        lines = fobj.readlines()

    expansion_list = []
    reconductor_list = []

    for this_line in lines:
        this_line = this_line.strip().split("\t")
        if 'exp' in this_line[0] and float(this_line[1]) > 0:
            temp_str = this_line[0].rstrip(']').split('[')
            temp_str = temp_str[1].split(',')
            expansion_list.append(temp_str[0] + '\t' + temp_str[1] + '\n')
        elif 'rec' in this_line[0] and float(this_line[1]) > 0:
            temp_str = this_line[0].rstrip(']').split('[')
            temp_str = temp_str[1].split(',')
            reconductor_list.append(temp_str[0] + ' \t ' + temp_str[1] + '\n')

    output_file_exp = 'TEPCE Results/TEPCE_summer_scaled_expansion_results.txt'
    output_file_rec = 'TEPCE Results/TEPCE_summer_scaled_recond_results.txt'


    with open(output_file_rec, 'w') as fobj:
        for line in reconductor_list:
            fobj.write(line)

    with open(output_file_exp, 'w') as fobj:
        for line in expansion_list:
            fobj.write(line)

def get_coords():
    sheet_name = 'SubStationsAZ_TableToExcel.xls'
    coords = pd.read_excel(sheet_name,  usecols='A,O,P')
    print(coords)

    input_file_exp = 'TEPCE Results/TEPCE_summer_scaled_expansion_results.txt'
    output_file_exp_coords = 'TEPCE Results/TEPCE_summer_scaled_expansion_coords.xls'

    with open(input_file_exp, 'r') as fobj:
        lines = fobj.readlines()

    exp_lines = {'from_x':[], 'from_y':[], 'to_x':[], 'to_y':[]}
    for this_line in lines:
        this_line = this_line.split()
        from_bus = int(this_line[0])
        to_bus = int(this_line[-1])
        exp_lines['from_y'].append(coords['LATITUDE'][from_bus])
        exp_lines['from_x'].append(coords['LONGITUDE'][from_bus])
        exp_lines['to_y'].append(coords['LATITUDE'][to_bus])
        exp_lines['to_x'].append(coords['LONGITUDE'][to_bus])
    exp_lines_data = pd.DataFrame.from_dict(exp_lines)
    exp_lines_data.to_excel(output_file_exp_coords)

    input_file_rec = 'TEPCE Results/TEPCE_summer_scaled_recond_results.txt'
    output_file_rec_coords = 'TEPCE Results/TEPCE_summer_scaled_rec_coords.xls'

    with open(input_file_rec, 'r') as fobj:
        lines = fobj.readlines()

    rec_lines = {'from_x': [], 'from_y': [], 'to_x': [], 'to_y': []}
    for this_line in lines:
        this_line = this_line.split()
        from_bus = int(this_line[0])
        to_bus = int(this_line[-1])
        rec_lines['from_y'].append(coords['LATITUDE'][from_bus])
        rec_lines['from_x'].append(coords['LONGITUDE'][from_bus])
        rec_lines['to_y'].append(coords['LATITUDE'][to_bus])
        rec_lines['to_x'].append(coords['LONGITUDE'][to_bus])
    rec_lines_data = pd.DataFrame.from_dict(rec_lines)
    rec_lines_data.to_excel(output_file_rec_coords)



def compare_outputs(file1, file2):
    with open(file1, 'r') as fobj:
        lines1 = fobj.readlines()
    with open(file2, 'r') as fobj:
        lines2 = fobj.readlines()

    set1 = set()
    set2 = set()
    for this_line in lines1:
        component = this_line.split()
        component = (int(component[0]), int(component[1]))
        set1.add(component)
    for this_line in lines2:
        component = this_line.split()
        component = (int(component[0]), int(component[1]))
        set2.add(component)

    intersect = set1.intersection(set2)
    symmetric_diff = set1.symmetric_difference(set2)
    two_not_one = set2.difference(set1)
    one_not_two = set1.difference(set2)

    filename = "pglib-opf-master/az_2022_case892.m"
    cap = 1
    demand = 1
    gen = 1
    gen_cost = .001

    bus_data = np.array([])
    gen_data = np.array([])
    branch_data = np.array([])
    bus_data, gen_data, branch_data = mp.load_data(filename)

    graph = nx.Graph()
    mp.encode_graph(graph, bus_data, gen_data, branch_data, demand, cap, gen, gen_cost)
    zone_counts1 = {}
    for node in graph.nodes:
        if graph.nodes[node]['bus_zone'] not in zone_counts1:
            zone_counts1[graph.nodes[node]['bus_zone']] = 0
    for component in set1:
        zone_counts1[max(graph.nodes[component[0]]['bus_zone'], graph.nodes[component[1]]['bus_zone'])] += 1
    zone_counts2 = {}
    for node in graph.nodes:
        if graph.nodes[node]['bus_zone'] not in zone_counts2:
            zone_counts2[graph.nodes[node]['bus_zone']] = 0
    for component in set2:
        zone_counts2[max(graph.nodes[component[0]]['bus_zone'], graph.nodes[component[1]]['bus_zone'])] += 1

    out_dict = {'intersect':intersect, 'symmetric_difference':symmetric_diff, 'one_not_two':one_not_two, 'two_not_one':two_not_one, 'zone_count1':zone_counts1.values(),'zone_count2':zone_counts2.values()}

    if 'exp' in file1:
        type1 = 'expansion'
        if 'simple' in file1:
            type1 += '_simple'
        if 'scale' in file1:
            type1 += '_summer_scale'
        if 'basic' in file1:
            type1 += '_basic'
        elif 'summer' in file1:
            type1 += '_summer'
    elif 'rec' in file1:
        type1 = 'reconductor'
        if 'simple' in file1:
            type1 += '_simple'
        if 'scale' in file1:
            type1 += '_summer_scale'
        if 'basic' in file1:
            type1 += '_basic'
        elif 'summer' in file1:
            type1 += '_summer'

    if 'exp' in file2:
        type2 = 'expansion'
        if 'simple' in file2:
            type2 += '_simple'
        if 'scale' in file2:
            type2 += '_summer_scale'
        if 'basic' in file2:
            type2 += '_basic'
        elif 'summer' in file2:
            type2 += '_summer'
    elif 'rec' in file2:
        type2 = 'reconductor'
        if 'simple' in file2:
            type2 += '_simple'
        if 'scale' in file2:
            type2 += '_summer_scale'
        if 'basic' in file2:
            type2 += '_basic'
        elif 'summer' in file2:
            type2 += '_summer'

    out_file = 'TEPCE Results/' + type1 + type2 + '.xls'

    out_panda = pd.DataFrame.from_dict(out_dict, orient='index')
    out_panda.to_excel(out_file)

get_bus_pairs()
get_coords()

results =['TEPCE Results/TEPCE' + filename.lstrip('./TEPCE Results\\') for filename in glob.glob('./TEPCE Results/*results.txt')]
#print(results)
results = list(itertools.product(results, results))

for pair in results:
    compare_outputs(pair[0],pair[1])



def graph(input_file, line_set):
    sheet_name = 'SubStationsAZ_TableToExcel.xls'
    coords = pd.read_excel(sheet_name,  usecols='A,O,P')
    geometry = gp.points_from_xy(coords['LONGITUDE'],coords['LATITUDE'])

    input_file_exp = input_file

    exp_frame = pd.read_excel(input_file_exp).transpose()
    exp_frame = exp_frame.rename(columns={0:'intersect',1:'sym_diff',2:'one-two',3:'two-one',4:'zone_count1',5:'zone_count2'})
    exp_frame = exp_frame.drop('Unnamed: 0')

    exp_lines = {'from_x': [], 'from_y': [], 'to_x': [], 'to_y': []}
    for row in exp_frame[line_set]:
        if len(str(row)) > 3:
            row = row.lstrip('(').rstrip(')').split(', ')
            from_bus = int(row[0])
            to_bus = int(row[-1])
            exp_lines['from_y'].append(coords['LATITUDE'][from_bus])
            exp_lines['from_x'].append(coords['LONGITUDE'][from_bus])
            exp_lines['to_y'].append(coords['LATITUDE'][to_bus])
            exp_lines['to_x'].append(coords['LONGITUDE'][to_bus])

    intersect_lines = []
    for row in range(len(exp_lines['from_x'])):
        line = LineString([(exp_lines['from_x'][row],exp_lines['from_y'][row]),(exp_lines['to_x'][row],exp_lines['to_y'][row])])
        intersect_lines.append(line)

    intersect_lines_shp = gp.GeoDataFrame(geometry=intersect_lines)

    states = gp.read_file('clustered_zones.shp')
    zones = pd.Series([1,5,5,5,4,3,3,3,2,2,2], index = range(11))
    states = states.assign(zone=zones)

    if 'exp' in input_file:
        compare = input_file.rstrip('.xls').split('expansion_')
        type = 'expansion_' + compare[1] + '_vs_' + compare[2]
    elif 'rec' in input_file:
        compare = input_file.rstrip('.xls').split('reconductor_')
        type = 'reonductor_' + compare[1] + '_vs_' + compare[2]

    light_jet = cmap_map(lambda x: x / 2 + 0.5, matplotlib.cm.Greys_r)
    ax = states.plot(column='zone', cmap=light_jet, figsize=(30,30))
    ax.set_axis_off()
    gdf = gp.GeoDataFrame(coords, geometry = geometry)

    filename = "pglib-opf-master/az_2022_case892.m"
    cap = 1
    demand = 1
    gen = 1
    gen_cost = .001

    bus_data = np.array([])
    gen_data = np.array([])
    branch_data = np.array([])
    bus_data, gen_data, branch_data = mp.load_data(filename)

    graph = nx.Graph()
    mp.encode_graph(graph, bus_data, gen_data, branch_data, demand, cap, gen, gen_cost)
    gdf.plot(ax=ax, markersize = [1 * val for val in nx.get_node_attributes(graph,'bus_pd').values()], color='none', edgecolor='midnightblue', label='Demand')
    gdf.plot(ax=ax, markersize = [1 * val for val in nx.get_node_attributes(graph, 'gen_Pmax').values()], color='none', edgecolor='fuchsia', label='Generation')
    intersect_lines_shp.plot(ax=ax, color='lightseagreen', alpha=0.4, label = 'Line Investment', linewidth=2)
    title = type.replace('_', ' ') + ' ' + line_set
    title = title.title()
    plt.title(title, size=48, weight = 'bold')
    legend = plt.legend(labelspacing=1, loc='lower left', bbox_to_anchor=(0.04,0.04), fontsize=14)
    for legend_handle in legend.legendHandles:
        legend_handle._sizes = [400]


    plt.savefig('TEPCE_Charts/' + type + '_' + line_set +'.png')



def graph_both(input_file, line_set):
    sheet_name = 'SubStationsAZ_TableToExcel.xls'
    coords = pd.read_excel(sheet_name,  usecols='A,O,P')
    geometry = gp.points_from_xy(coords['LONGITUDE'],coords['LATITUDE'])

    input_file_rec = input_file[0]
    input_file_exp = input_file[1]

    exp_frame = pd.read_excel(input_file_exp).transpose()
    exp_frame = exp_frame.rename(columns={0:'intersect',1:'sym_diff',2:'one-two',3:'two-one',4:'zone_count1',5:'zone_count2'})
    exp_frame = exp_frame.drop('Unnamed: 0')

    exp_lines = {'from_x': [], 'from_y': [], 'to_x': [], 'to_y': []}
    for row in exp_frame[line_set]:
        if len(str(row)) > 3:
            row = row.lstrip('(').rstrip(')').split(', ')
            from_bus = int(row[0])
            to_bus = int(row[-1])
            exp_lines['from_y'].append(coords['LATITUDE'][from_bus])
            exp_lines['from_x'].append(coords['LONGITUDE'][from_bus])
            exp_lines['to_y'].append(coords['LATITUDE'][to_bus])
            exp_lines['to_x'].append(coords['LONGITUDE'][to_bus])

    exp_intersect_lines = []
    for row in range(len(exp_lines['from_x'])):
        line = LineString([(exp_lines['from_x'][row],exp_lines['from_y'][row]),(exp_lines['to_x'][row],exp_lines['to_y'][row])])
        exp_intersect_lines.append(line)

    exp_intersect_lines_shp = gp.GeoDataFrame(geometry=exp_intersect_lines)

    rec_frame = pd.read_excel(input_file_rec).transpose()
    rec_frame = rec_frame.rename(columns={0: 'intersect', 1: 'sym_diff', 2: 'one-two', 3: 'two-one', 4: 'zone_count1', 5: 'zone_count2'})
    rec_frame = rec_frame.drop('Unnamed: 0')

    rec_lines = {'from_x': [], 'from_y': [], 'to_x': [], 'to_y': []}
    for row in rec_frame[line_set]:
        if len(str(row)) > 3:
            row = row.lstrip('(').rstrip(')').split(', ')
            from_bus = int(row[0])
            to_bus = int(row[-1])
            rec_lines['from_y'].append(coords['LATITUDE'][from_bus])
            rec_lines['from_x'].append(coords['LONGITUDE'][from_bus])
            rec_lines['to_y'].append(coords['LATITUDE'][to_bus])
            rec_lines['to_x'].append(coords['LONGITUDE'][to_bus])

    rec_intersect_lines = []
    for row in range(len(rec_lines['from_x'])):
        line = LineString([(rec_lines['from_x'][row], rec_lines['from_y'][row]), (rec_lines['to_x'][row], rec_lines['to_y'][row])])
        rec_intersect_lines.append(line)

    rec_intersect_lines_shp = gp.GeoDataFrame(geometry=rec_intersect_lines)

    states = gp.read_file('clustered_zones.shp')
    zones = pd.Series([1,3,3,3,4,5,5,5,2,2,2], index = range(11))
    states = states.assign(zone=zones)

    exp_compare = input_file_exp.rstrip('.xls').split('expansion_')
    exp_type = 'expansion_this'
    try:
        exp_type = 'expansion_' + exp_compare[1] + '_vs_' + exp_compare[2]
    except:
        pass
    rec_compare = input_file_rec.rstrip('.xls').split('reconductor_')
    rec_type = 'reconductor_this'
    try:
        rec_type = 'reonductor_' + rec_compare[1] + '_vs_' + rec_compare[2]
    except:
        pass

    light_jet = cmap_map(lambda x: x / 2 + 0.5, matplotlib.cm.Greys_r)
    ax = states.plot(column='zone', cmap=light_jet, figsize=(30,30))
    ax.set_axis_off()
    gdf = gp.GeoDataFrame(coords, geometry = geometry)

    filename = "pglib-opf-master/az_2022_case892.m"
    cap = 1
    demand = 1
    gen = 1
    gen_cost = .001

    bus_data = np.array([])
    gen_data = np.array([])
    branch_data = np.array([])
    bus_data, gen_data, branch_data = mp.load_data(filename)

    graph = nx.Graph()
    mp.encode_graph(graph, bus_data, gen_data, branch_data, demand, cap, gen, gen_cost)
    gdf.plot(ax=ax, markersize = [1 * val for val in nx.get_node_attributes(graph,'bus_pd').values()], color='none', edgecolor='midnightblue', label='Demand')
    gdf.plot(ax=ax, markersize = [1 * val for val in nx.get_node_attributes(graph, 'gen_Pmax').values()], color='none', edgecolor='fuchsia', label='Generation')
    exp_intersect_lines_shp.plot(ax=ax, color='lightseagreen', alpha=0.6, label = 'Expansion Investment', linewidth=3)
    rec_intersect_lines_shp.plot(ax=ax, color='rebeccapurple', alpha=0.6, label='Reconductoring Investment', linewidth=3)
    title = exp_type.replace('_', ' ') + ' ' + line_set
    title = title[10:]
    #print(title)
    title = title.title()
    #print(title)
    title = title.replace('Summer Scale Summer', 'Experiment 3')
    #print(title)
    title = title.replace('Basic','Experiment 0')
    #print(title)
    title = title.replace('Simple', 'Experiment 1')
    #print(title)
    title = title.replace('Summer', 'Experiment 2')
    #print(title)
    if 'One-Two' in title:
        title = title.replace('Vs', 'Without')
        title = title.replace(' One-Two','')
    elif 'Two-One' in title:
        title = title.replace('Vs','Without')
        title = title.replace(' Two-One', '')
        x = title[:12]
        y = title[21:]
        title = y + ' Without ' + x
    elif 'Sym_Diff' in title:
        title = title.replace('Vs', 'Symmetric Difference')
        title = title.replace(' Sym_Diff', '')
    elif 'Intersect' in title:
        title = title.replace(' Intersect', '')
        title = title.replace('Vs', 'Intersect')


    plt.title(title, size=48, weight = 'bold')
    #legend = plt.legend(labelspacing=1, loc='lower left', bbox_to_anchor=(0.04,0.04), fontsize=14)
    #for legend_handle in legend.legendHandles:
    #    legend_handle._sizes = [400]

    #label_params = ax.get_legend_handles_labels()
    #figl, axl = plt.subplots()
    #axl.axis(False)
    #axl.legend(*label_params, loc="center", bbox_to_anchor=(0.5, 0.5), prop={"size": 27},frameon=False)
    #figl.savefig("LABEL_ONLY.png")

    plt.savefig('TEPCE_Charts/Exp_Rec/' + exp_type + '_' + line_set +'.png')


def plot_only():
    columns = {0: 'intersect', 1: 'sym_diff', 2: 'one-two', 3: 'two-one'}
    recond_compare = ['TEPCE Results/' + filename.lstrip('./TEPCE Results\\') for filename in glob.glob('./TEPCE Results/reconductor*.xls')]
    expand_compare = ['TEPCE Results/e' + filename.lstrip('./TEPCE Results\\') for filename in glob.glob('./TEPCE Results/expansion*.xls')]
    xls_list = list(zip(recond_compare, expand_compare))
    xls_list = [item for item in xls_list if 'sca' in item[0] and 'le' not in item[0]]
    flag = '12'
    print(xls_list)
    #xls_list = xls_list[4:]
    #print(xls_list)
    #xls_list = xls_list[4:]
    print(xls_list)


    sheet_name = 'SubStationsAZ_TableToExcel.xls'
    coords = pd.read_excel(sheet_name,  usecols='A,O,P')
    geometry = gp.points_from_xy(coords['LONGITUDE'],coords['LATITUDE'])

    final_exp_list = set()
    final_rec_list = set()

    line_set = 'one-two'

    for input_file in xls_list:
        input_file_rec = input_file[0]
        input_file_exp = input_file[1]

        exp_frame = pd.read_excel(input_file_exp).transpose()
        exp_frame = exp_frame.rename(columns={0:'intersect',1:'sym_diff',2:'one-two',3:'two-one',4:'zone_count1',5:'zone_count2'})
        exp_frame = exp_frame.drop('Unnamed: 0')

        exp_lines = {'from_x': [], 'from_y': [], 'to_x': [], 'to_y': []}
        for row in exp_frame[line_set]:
            if len(str(row)) > 3:
                row = row.lstrip('(').rstrip(')').split(', ')
                from_bus = int(row[0])
                to_bus = int(row[-1])
                exp_lines['from_y'].append(coords['LATITUDE'][from_bus])
                exp_lines['from_x'].append(coords['LONGITUDE'][from_bus])
                exp_lines['to_y'].append(coords['LATITUDE'][to_bus])
                exp_lines['to_x'].append(coords['LONGITUDE'][to_bus])

        exp_intersect_lines = []
        for row in range(len(exp_lines['from_x'])):
            line = ((exp_lines['from_x'][row],exp_lines['from_y'][row]),(exp_lines['to_x'][row],exp_lines['to_y'][row]))
            #line = LineString([(exp_lines['from_x'][row],exp_lines['from_y'][row]),(exp_lines['to_x'][row],exp_lines['to_y'][row])])
            exp_intersect_lines.append(line)

        if len(final_exp_list) == 0:
            final_exp_list = set(exp_intersect_lines)
        else:
            final_exp_list &= set(exp_intersect_lines)

        rec_frame = pd.read_excel(input_file_rec).transpose()
        rec_frame = rec_frame.rename(columns={0: 'intersect', 1: 'sym_diff', 2: 'one-two', 3: 'two-one', 4: 'zone_count1', 5: 'zone_count2'})
        rec_frame = rec_frame.drop('Unnamed: 0')

        rec_lines = {'from_x': [], 'from_y': [], 'to_x': [], 'to_y': []}
        for row in rec_frame[line_set]:
            if len(str(row)) > 3:
                row = row.lstrip('(').rstrip(')').split(', ')
                from_bus = int(row[0])
                to_bus = int(row[-1])
                rec_lines['from_y'].append(coords['LATITUDE'][from_bus])
                rec_lines['from_x'].append(coords['LONGITUDE'][from_bus])
                rec_lines['to_y'].append(coords['LATITUDE'][to_bus])
                rec_lines['to_x'].append(coords['LONGITUDE'][to_bus])

        rec_intersect_lines = []
        for row in range(len(rec_lines['from_x'])):
            line = ((rec_lines['from_x'][row], rec_lines['from_y'][row]), (rec_lines['to_x'][row], rec_lines['to_y'][row]))
            #line = LineString([(rec_lines['from_x'][row], rec_lines['from_y'][row]), (rec_lines['to_x'][row], rec_lines['to_y'][row])])
            rec_intersect_lines.append(line)

        if len(final_rec_list) == 0:
            final_rec_list = set(rec_intersect_lines)
        else:
            final_rec_list &= set(rec_intersect_lines)

    for item in final_exp_list:
        print(item)

    final_exp_list = [LineString([item[0],item[1]]) for item in final_exp_list]
    final_rec_list = [LineString([item[0],item[1]]) for item in final_rec_list]

    rec_intersect_lines_shp = gp.GeoDataFrame(geometry=final_rec_list)
    exp_intersect_lines_shp = gp.GeoDataFrame(geometry=final_exp_list)
    states = gp.read_file('clustered_zones.shp')
    zones = pd.Series([1, 3, 3, 3, 4, 5, 5, 5, 2, 2, 2], index=range(11))
    states = states.assign(zone=zones)

    light_jet = cmap_map(lambda x: x / 2 + 0.5, matplotlib.cm.Greys_r)
    ax = states.plot(column='zone', cmap=light_jet, figsize=(30, 30))
    ax.set_axis_off()
    gdf = gp.GeoDataFrame(coords, geometry=geometry)

    filename = "pglib-opf-master/az_2022_case892.m"
    cap = 1
    demand = 1
    gen = 1
    gen_cost = .001

    bus_data = np.array([])
    gen_data = np.array([])
    branch_data = np.array([])
    bus_data, gen_data, branch_data = mp.load_data(filename)

    graph = nx.Graph()
    mp.encode_graph(graph, bus_data, gen_data, branch_data, demand, cap, gen, gen_cost)
    gdf.plot(ax=ax, markersize=[1 * val for val in nx.get_node_attributes(graph, 'bus_pd').values()], color='none', edgecolor='midnightblue', label='Demand')
    gdf.plot(ax=ax, markersize=[1 * val for val in nx.get_node_attributes(graph, 'gen_Pmax').values()], color='none', edgecolor='fuchsia', label='Generation')
    exp_intersect_lines_shp.plot(ax=ax, color='lightseagreen', alpha=0.6, label='Expansion Investment', linewidth=3)
    rec_intersect_lines_shp.plot(ax=ax, color='rebeccapurple', alpha=0.6, label='Reconductoring Investment', linewidth=3)

    title = 'Experiment 3 Only'
    plt.title(title, size=48, weight='bold')

    plt.savefig('TEPCE_Charts/Exp_Rec/experiment_4_without_' + flag + '.png')


################################
######## Main Operations #######
################################

columns={0:'intersect',1:'sym_diff',2:'one-two',3:'two-one'}
recond_compare = ['TEPCE Results/' + filename.lstrip('./TEPCE Results\\') for filename in glob.glob('./TEPCE Results/reconductor*.xls')]
expand_compare = ['TEPCE Results/e' + filename.lstrip('./TEPCE Results\\') for filename in glob.glob('./TEPCE Results/expansion*.xls')]
print(recond_compare)
print(expand_compare)
xls_list = list(zip(recond_compare,expand_compare))


for value in columns.values():
    for filename in xls_list:
        graph_both(filename, value)
        #graph(filename,value)
plot_only()

'''for value in columns.values():
    for filename in xls_list:
        graph_both(filename, value)
        #graph(filename,value)'''



################################################
####### Graph Weather Station Location #########
################################################

sheetname = 'weatherstations.xls'
coords = pd.read_excel(sheetname,  usecols='A,B,C')

geometry = gp.points_from_xy(coords['longitude'],coords['latitude'])
states = gp.read_file('clustered_zones.shp')
ax = states.plot(figsize=(30,30), color = 'none')
ax.set_axis_off()
gdf = gp.GeoDataFrame(coords, geometry = geometry)
print(coords)
gdf.plot(ax=ax, c = coords['cluster'], markersize = 1000)
from matplotlib.lines import Line2D
cmap = plt.cm.viridis
custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4),
                Line2D([0], [0], color=cmap(.25), lw=4),
                Line2D([0], [0], color=cmap(.5), lw=4),
                Line2D([0], [0], color=cmap(.75), lw=4),
                Line2D([0], [0], color=cmap(1.), lw=4)]
legend = ax.legend(custom_lines, ['Region 1', 'Region 2', 'Region 3', 'Region 4', 'Region 5'], loc='lower left',bbox_to_anchor=(0,0.02), fontsize=32)
#plt.show()
plt.savefig('weatherstations.png')



'''sheetname = 'weatherstations.xls'
coords = pd.read_excel(sheetname,  usecols='A,B,C')

geometry = gp.points_from_xy(coords['longitude'],coords['latitude'])
states = gp.read_file('clustered_zones.shp')
zones = pd.Series([1,3,3,3,4,5,5,5,2,2,2], index = range(11))
zone_names = pd.Series(['1','5','5','5','4','3','3','3','2','2','2'], index = range(11))
who_knows = list(zip(zone_names,zones))
states = states.assign(zone=zones)
ax = states.plot(figsize=(30,30), column = 'zone')
ax.set_axis_off()
gdf = gp.GeoDataFrame(coords, geometry = geometry)
print(states)
title = "Temperature Regions by Clustering Assignment"
plt.title(title, size=48, weight = 'bold')
#legend = plt.legend(labelspacing=1, loc='lower left', fontsize=14)
#handles = legend.legendHandles
#print(handles)
#for legend_handle in legend.legendHandles:
#    legend_handle._sizes = [400]
from matplotlib.lines import Line2D
cmap = plt.cm.viridis
custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4),
                Line2D([0], [0], color=cmap(.25), lw=4),
                Line2D([0], [0], color=cmap(.5), lw=4),
                Line2D([0], [0], color=cmap(.75), lw=4),
                Line2D([0], [0], color=cmap(1.), lw=4)]
legend = ax.legend(custom_lines, ['Region 1', 'Region 2', 'Region 3', 'Region 4', 'Region 5'], loc='lower left',bbox_to_anchor=(0,0.02), fontsize=32)

#plt.show()
plt.savefig('weatherzones.png')'''