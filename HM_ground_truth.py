import plotly.graph_objects as go
import pandas as pd
import numpy as np
import datetime


alldata = pd.read_csv("classification_data_ts.csv")
# Drop all rows where HM column is not null
alldata = alldata[alldata['HM'].notna()]
# Where row index = 300, change HM value to 1
alldata.loc[300, 'HM'] = 1
# Where RepID = 1489, change HM value to 0
alldata.loc[alldata['RepID'] == 1489, 'HM'] = 0
# Where HM = -1, change HM value to 0
alldata.loc[alldata['HM'] == -1, 'HM'] = 0

plotdf = alldata.reindex(columns=['DOY', 'N2OFlux', 'RepID'])

id_list = plotdf['RepID'].unique()
id_list = [1489]
for id_iter in id_list:
    flux_list = list(plotdf.loc[(plotdf['RepID'] == id_iter)].N2OFlux)
    doy_list = list(plotdf.loc[(plotdf['RepID'] == id_iter)].DOY)

    f = go.FigureWidget([go.Scatter(x=doy_list, y=flux_list, mode='markers')])

    scatter = f.data[0]

    colors = ['#a3a7e4'] * 100
    scatter.marker.color = colors
    scatter.marker.size = [10] * 100
    f.layout.hovermode = 'closest'



    # create our callback function
    def update_point(trace, points, selector):
        c = list(scatter.marker.color)
        s = list(scatter.marker.size)
        for i in points.point_inds:
            c[i] = '#bae2be'
            s[i] = 20
            with f.batch_update():
                scatter.marker.color = c
                scatter.marker.size = s


    scatter.on_click(update_point)

    f