from __future__ import print_function
import pandas as pd
import statistics
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import os
import shutil
import tkinter as tk
from tkinter import *
from tkcalendar import *
from tkinter import ttk
from src.data.db_conn import load_db_table

class gn_formatter():
    def main(self):
        gn_filename = "GN_Master/natres.csv"
        gracedf = pd.read_csv(gn_filename)
        gracedf['Date'] = pd.to_datetime(gracedf['Date'])
        gracedf.sort_values(by='Date', inplace=True)
        weatherdf = pd.read_csv("GN_Master/PrecipDaily.csv")
        gn_formatter.formatter(gracedf, weatherdf)

    def formatter(self, gracedf, weatherdf):
        print(gracedf.head())
        sites = gracedf['SiteID'].unique()
        print(sites)
        for site in sites:
            sitedf = gracedf.query('SiteID == @site')
            treatments = sitedf['TreatmentID'].unique()
            siteweather = weatherdf.query('SiteID == @site')
            precipdf = siteweather[['Weather Date', 'Precip mm/d']]
            try:
                os.mkdir("GRACEnet/" + site)
            except FileExistsError:
                # directory already exists
                pass
            for treatment in treatments:
                try:
                    os.mkdir("GRACEnet/" + site + "/" + treatment)
                except FileExistsError:
                    # directory already exists
                    pass
                treatmentdf = sitedf.query('TreatmentID == @treatment')
                treatmentdf.sort_values(by='Date', inplace=True)

                treatmentdf["Delta"] = treatmentdf['Date'].diff()[1:]
                checkdf = treatmentdf[['Date', 'TreatmentID', 'Delta']]
                print(checkdf.head())

                reps = treatmentdf['ExpUnitID'].unique()
                for rep in reps:
                    print(rep)
                    try:
                        os.mkdir("GRACEnet/" + site + "/" + treatment + "/" + rep)
                    except FileExistsError:
                        # directory already exists
                        pass
                    repdf = treatmentdf.query('ExpUnitID == @rep')
                    print(repdf)
                    # Check for large gaps in dates to split into separate time series if necessary
                    gapnum = repdf[repdf.Delta > pd.Timedelta('30 days')].shape[0]
                    print(gapnum)

                    # If there are large gaps in the time series, split these into individual time series
                    if gapnum > 0:
                        gapdf = repdf[repdf.Delta > pd.Timedelta('30 days')]
                        print(gapdf)
                        i = 1
                        startindex = repdf.index[0]
                        lastindex = repdf.index[-1]

                        for gapindex in gapdf.index:
                            try:
                                os.mkdir("GRACEnet/" + site + "/" + treatment + "/" + rep + "/" + str(i))
                            except FileExistsError:
                                # directory already exists
                                pass
                            print(startindex)
                            print(gapindex)
                            continuousdf = repdf.loc[int(startindex) : int(gapindex) - 1]
                            print(continuousdf)
                            fluxdf = continuousdf[['Date', 'N2O gN/ha/d']]
                            airtdf = continuousdf[['Date', 'Air Temp degC']]
                            soiltdf = continuousdf[['Date', 'Soil Temp degC']]
                            vwcdf = continuousdf[['Date', 'Soil Moisture % vol']]

                            fluxdf.to_csv("GRACEnet/" + site + "/" + treatment + "/" + rep + "/" + str(i) + "/" + 'flux.csv')
                            airtdf.to_csv("GRACEnet/" + site + "/" + treatment + "/" + rep + "/" + str(i) + "/" + 'air temp.csv')
                            soiltdf.to_csv("GRACEnet/" + site + "/" + treatment + "/" + rep + "/" + str(i) + "/" + 'soil temp.csv')
                            vwcdf.to_csv("GRACEnet/" + site + "/" + treatment + "/" + rep + "/" + str(i) + "/" + 'vwc.csv')
                            precipdf.to_csv("GRACEnet/" + site + "/" + treatment + "/" + rep + "/" + str(i) + "/" + 'precip.csv')
                            startindex = gapindex
                            i += 1

                        try:
                            os.mkdir("GRACEnet/" + site + "/" + treatment + "/" + rep + "/" + str(i))
                        except FileExistsError:
                            # directory already exists
                            pass
                        print(gapindex)
                        print(lastindex)
                        continuousdf = repdf.loc[int(gapindex): int(lastindex) ]
                        fluxdf = continuousdf[['Date', 'N2O gN/ha/d']]
                        airtdf = continuousdf[['Date', 'Air Temp degC']]
                        soiltdf = continuousdf[['Date', 'Soil Temp degC']]
                        vwcdf = continuousdf[['Date', 'Soil Moisture % vol']]
                        print(continuousdf)
                        print(fluxdf)

                        fluxdf.to_csv("GRACEnet/" + site + "/" + treatment + "/" + rep + "/" + str(i) + "/" +  'flux.csv')
                        airtdf.to_csv("GRACEnet/" + site + "/" + treatment + "/" + rep + "/" + str(i) + "/" + 'air temp.csv')
                        soiltdf.to_csv("GRACEnet/" + site + "/" + treatment + "/" + rep + "/" + str(i) + "/" + 'soil temp.csv')
                        vwcdf.to_csv("GRACEnet/" + site + "/" + treatment + "/" + rep + "/" + str(i) + "/" + 'vwc.csv')
                        precipdf.to_csv(
                            "GRACEnet/" + site + "/" + treatment + "/" + rep + "/" + str(i) + "/" + 'precip.csv')
                        gapdf.query()

                    # Else, process the entire time series
                    else:
                        fluxdf = repdf[['Date', 'N2O gN/ha/d']]
                        airtdf = repdf[['Date', 'Air Temp degC']]
                        soiltdf = repdf[['Date', 'Soil Temp degC']]
                        vwcdf = repdf[['Date', 'Soil Moisture % vol']]
                        print(rep)
                        print(fluxdf)
                        fluxdf.to_csv("GRACEnet/" + site + "/" + treatment + "/" + rep + "/" + 'flux.csv')
                        airtdf.to_csv("GRACEnet/" + site + "/" + treatment + "/" + rep + "/" + 'air temp.csv')
                        soiltdf.to_csv("GRACEnet/" + site + "/" + treatment + "/" + rep + "/" + 'soil temp.csv')
                        vwcdf.to_csv("GRACEnet/" + site + "/" + treatment + "/" + rep + "/" + 'vwc.csv')
                        precipdf.to_csv("GRACEnet/" + site + "/" + treatment + "/" + rep + "/" + 'precip.csv')
                        print("GRACEnet/" + site + "/" + treatment + "/" + rep + "/" + 'flux.csv')



if __name__ == '__main__':
    gn_formatter = gn_formatter()
    gn_formatter.main()