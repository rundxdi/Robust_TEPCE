import pandas as pd

import geopandas as gp
import matplotlib.pyplot as plt
import contextily as cx


sheetname = 'weatherstations.xls'
coords = pd.read_excel(sheetname,  usecols='A,B,C')



geometry = gp.points_from_xy(coords['longitude'],coords['latitude'])
states = gp.read_file('clustered_zones.shp')
ax = states.plot(figsize=(30,30), color = 'none')
#df.plot(ax=ax)
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
plt.show()
plt.savefig('weatherstations.png')