import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from ast import literal_eval
import re

def read_kyles_awful_data_format(target_file):
    with open(target_file, "r") as fobj:
        lines = fobj.readlines()
    # scrape info from filename with regex
    regex_scarper = re.compile('output_results_bus(.*)_line(.*)')
    scrape = regex_scarper.findall(target_file.stem)[0]

    # great dict for data
    return_dict = {"bus fail percent": literal_eval(scrape[0])*100, "line fail percent": literal_eval(scrape[1])*100}
    for this_line in lines:
        this_line = this_line.strip().split("\t")
        return_dict[this_line[0]] = literal_eval(this_line[1]) # NOT SAFE AT ALL

    # post-processing on load shed
    return_dict["shed (MW)"] = (return_dict["shed (MW)"] - 329200)/100
    return return_dict

# read data
target_dir = Path("./OTS_Results") # CHANGE THIS TO YOUR TARGET DIR
data_dicts = []

# go through each file, pull into a dict
for target_file in target_dir.glob("*"):
    print(target_file)
    data_dicts.append(read_kyles_awful_data_format(target_file))


# peel out keys for matrix axis
bus_keys = set()
line_keys = set()
for item in data_dicts:
    bus_keys.add(item["bus fail percent"])
    line_keys.add(item["line fail percent"])

# this is SUPER inefficicent but of well

# allocate matrix
load_shed_matrix = np.empty((len(bus_keys), len(line_keys)))
bus_investment_matrix = np.empty((len(bus_keys), len(line_keys)))
generator_investment_matrix = np.empty((len(bus_keys), len(line_keys)))
line_investment_matrix = np.empty((len(bus_keys), len(line_keys)))
line_percent_matrix = np.empty((len(bus_keys), len(line_keys)))
generator_investment_cost_matrix = np.empty((len(bus_keys), len(line_keys)))
line_investment_cost_matrix = np.empty((len(bus_keys), len(line_keys)))
bus_investment_cost_matrix = np.empty((len(bus_keys), len(line_keys)))
load_shed_matrix[:] = np.nan
bus_investment_matrix[:] = np.nan
generator_investment_matrix[:] = np.nan
line_investment_matrix[:] = np.nan
line_percent_matrix[:] = np.nan
generator_investment_cost_matrix[:] = np.nan
line_investment_cost_matrix[:] = np.nan
bus_investment_cost_matrix[:] = np.nan
# fill matrix
for ix, this_bus in enumerate(bus_keys):
    for iy, this_line in enumerate(line_keys):
        for this_dict in data_dicts:
            if (this_dict["bus fail percent"] == this_bus) and (this_dict["line fail percent"] == this_line):
                load_shed_matrix[ix, iy] = this_dict["shed (MW)"] - 29623
                bus_investment_matrix[ix, iy] = this_dict["investment"][0]
                generator_investment_matrix[ix, iy] = this_dict["investment"][1]
                line_investment_matrix[ix, iy] = this_dict["investment"][2]
                line_percent_matrix[ix,iy] = (this_dict["investment"][2]*5)/500
                generator_investment_cost_matrix[ix, iy] = this_dict["investment"][1]*10
                line_investment_cost_matrix[ix, iy] = this_dict["investment"][2]*5
                bus_investment_cost_matrix[ix, iy] = this_dict["investment"][0]

# plotting params
major_fontsize = 20
mid_fontsize = 16
minor_fontsize = 12

# print(load_shed_matrix)
plot_types = ["Percent Load Shed", "Bus Investment", "Generator Investment", "Line Investment", "Line Investment Percentage", "Generator Investment Cost", "Line Investment Cost", "Bus Investment Cost"]
units = ["MW", "Number of Investments", "Number of Investments", "Number of Investments", "Investment Percentage", "Cost of Investment (man-hours)", "Cost of Investment (man-hours)", "Cost of Investment (man-hours)"]
data = [load_shed_matrix, bus_investment_matrix, generator_investment_matrix, line_investment_matrix, line_percent_matrix, generator_investment_cost_matrix, line_investment_cost_matrix, bus_investment_cost_matrix]

# make all plots
for this_plot_type, this_matrix, this_unit in zip(plot_types, data, units):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    cax = ax.imshow(this_matrix, interpolation='nearest')
    cbar = fig.colorbar(cax)

    for (i, j), z in np.ndenumerate(this_matrix):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

    ax.set_xticklabels([''] + list(line_keys), fontsize=minor_fontsize)
    ax.set_yticklabels([''] + list(bus_keys), fontsize=minor_fontsize)
    ax.set_xlabel("Line Fail Percent", fontsize=mid_fontsize)
    ax.set_ylabel("Bus Fail Percent", fontsize=mid_fontsize)
    ax.set_title(this_plot_type, fontsize=major_fontsize)
    
    cbar.set_label(this_unit, fontsize=mid_fontsize)

    # plt.show()
    fig.savefig(f"{this_plot_type.lower().replace(' ', '_')}.png")
