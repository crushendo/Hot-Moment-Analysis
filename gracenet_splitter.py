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
        gracedf.sort_values(by=['ExpUnitID', 'Date'], inplace=True)
        gracedf.reset_index(inplace = True)
        del gracedf['index']
        weatherdf = pd.read_csv("GN_Master/PrecipDaily.csv")
        gn_formatter.formatter(gracedf, weatherdf)

    def formatter(self, gracedf, weatherdf):
        print(gracedf.head())
        sites = gracedf['SiteID'].unique()
        print(sites)
        for site in sites:
            print(site)
            sitedf = gracedf.query('SiteID == @site')
            treatments = sitedf['TreatmentID'].unique()
            siteweather = weatherdf.query('SiteID == @site')
            precipdf = siteweather[['Weather Date', 'Precip mm/d']]

            format1 = "%m/%d/%Y"
            format2 = "%Y-%m-%d"
            format3 = "%m/%d/%Y %H:%M"
            format4 = "%Y-%m-%d %H:%M"
            precipdf["Weather Date"] = precipdf["Weather Date"].apply(
                lambda x: "0" +  x if x[1] == "/" else x)
            precipdf["Weather Date"] = precipdf["Weather Date"].apply(
                lambda x: x[:3] + "0" + x[3:] if x[4] == "/" else x)
            precipdf["Weather Date"] = precipdf["Weather Date"].apply(
                lambda x: x[:10])
            print(precipdf)
            try:
                precipdf["Weather Date"] = precipdf["Weather Date"].apply(
                    lambda x: datetime.strptime(x, format1).strftime(format2))
            except:
                precipdf["Weather Date"] = precipdf["Weather Date"].apply(
                    lambda x: datetime.strptime(x, format3).strftime(format2))
            try:
                os.mkdir("GRACEnet/" + site)
            except FileExistsError:
                # directory already exists
                pass
            for treatment in treatments:
                if "/" in treatment:
                    print("here")
                    treatment = treatment.replace("/", ">", 5)
                    print(treatment)
                try:
                    print(treatment)
                    os.mkdir("GRACEnet/" + site + "/" + treatment)
                except FileExistsError:
                    # directory already exists
                    pass
                treatmentdf = sitedf.query('TreatmentID == @treatment')

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
                    repdf["Delta"] = repdf['Date'].diff()[1:]
                    checkdf = repdf[['Date', 'TreatmentID', 'Delta']]
                    print(checkdf.head())

                    # Check for large gaps in dates to split into separate time series if necessary
                    gapnum = repdf[repdf.Delta > pd.Timedelta('45 days')].shape[0]
                    print(gapnum)

                    # If there are large gaps in the time series, split these into individual time series
                    if gapnum > 0:
                        gapdf = repdf[repdf.Delta > pd.Timedelta('45 days')]
                        print(gapdf)
                        i = 1
                        startindex = repdf.index[0]
                        lastindex = repdf.index[-1]

                        for gapindex in gapdf.index:
                            if repdf.loc[int(startindex): int(gapindex) - 1].shape[0] < 5:
                                startindex = gapindex
                                continue
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

                            if fluxdf["N2O gN/ha/d"].notnull().sum() > 0:
                                fluxdf.to_csv(
                                    "GRACEnet/" + site + "/" + treatment + "/" + rep + "/" + str(i) + "/" + 'flux.csv',
                                index=False)
                            if airtdf["Air Temp degC"].notnull().sum() > 0:
                                airtdf.to_csv("GRACEnet/" + site + "/" + treatment + "/" + rep + "/" + str(
                                    i) + "/" + 'air temp.csv', index=False)
                            if soiltdf["Soil Temp degC"].notnull().sum() > 0:
                                soiltdf.to_csv("GRACEnet/" + site + "/" + treatment + "/" + rep + "/" + str(
                                    i) + "/" + 'soil temp.csv', index=False)
                            if vwcdf["Soil Moisture % vol"].notnull().sum() > 0:
                                vwcdf.to_csv(
                                    "GRACEnet/" + site + "/" + treatment + "/" + rep + "/" + str(i) + "/" + 'vwc.csv',
                                    index=False)
                            if precipdf["Precip mm/d"].notnull().sum() > 0:
                                precipdf.to_csv(
                                    "GRACEnet/" + site + "/" + treatment + "/" + rep + "/" + str(
                                        i) + "/" + 'precip.csv', index=False)
                            startindex = gapindex
                            i += 1

                        if repdf.loc[int(startindex): int(gapindex) - 1].shape[0] > 4:
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

                            if fluxdf["N2O gN/ha/d"].notnull().sum() > 0:
                                fluxdf.to_csv(
                                    "GRACEnet/" + site + "/" + treatment + "/" + rep + "/" + str(i) + "/" + 'flux.csv',
                                    index=False)
                            if airtdf["Air Temp degC"].notnull().sum() > 0:
                                airtdf.to_csv("GRACEnet/" + site + "/" + treatment + "/" + rep + "/" + str(
                                    i) + "/" + 'air temp.csv', index=False)
                            if soiltdf["Soil Temp degC"].notnull().sum() > 0:
                                soiltdf.to_csv("GRACEnet/" + site + "/" + treatment + "/" + rep + "/" + str(
                                    i) + "/" + 'soil temp.csv', index=False)
                            if vwcdf["Soil Moisture % vol"].notnull().sum() > 0:
                                vwcdf.to_csv(
                                    "GRACEnet/" + site + "/" + treatment + "/" + rep + "/" + str(i) + "/" + 'vwc.csv',
                                    index=False)
                            if precipdf["Precip mm/d"].notnull().sum() > 0:
                                precipdf.to_csv(
                                    "GRACEnet/" + site + "/" + treatment + "/" + rep + "/" + str(
                                        i) + "/" + 'precip.csv', index=False)
                            #gapdf.query()

                    # Else, process the entire time series
                    else:
                        fluxdf = repdf[['Date', 'N2O gN/ha/d']]
                        airtdf = repdf[['Date', 'Air Temp degC']]
                        soiltdf = repdf[['Date', 'Soil Temp degC']]
                        vwcdf = repdf[['Date', 'Soil Moisture % vol']]
                        print(rep)
                        print(fluxdf)
                        if fluxdf["N2O gN/ha/d"].notnull().sum() > 0:
                            fluxdf.to_csv("GRACEnet/" + site + "/" + treatment + "/" + rep + "/" + 'flux.csv',
                                          index=False)
                        if airtdf["Air Temp degC"].notnull().sum() > 0:
                            airtdf.to_csv("GRACEnet/" + site + "/" + treatment + "/" + rep + "/" + 'air temp.csv',
                                          index=False)
                        if soiltdf["Soil Temp degC"].notnull().sum() > 0:
                            soiltdf.to_csv("GRACEnet/" + site + "/" + treatment + "/" + rep + "/" + 'soil temp.csv',
                                           index=False)
                        if vwcdf["Soil Moisture % vol"].notnull().sum() > 0:
                            vwcdf.to_csv("GRACEnet/" + site + "/" + treatment + "/" + rep + "/" + 'vwc.csv',
                                         index=False)
                        if precipdf["Precip mm/d"].notnull().sum() > 0:
                            precipdf.to_csv("GRACEnet/" + site + "/" + treatment + "/" + rep + "/" + 'precip.csv',
                                            index=False)
                        print("GRACEnet/" + site + "/" + treatment + "/" + rep + "/" + 'flux.csv')



if __name__ == '__main__':
    gn_formatter = gn_formatter()
    gn_formatter.main()