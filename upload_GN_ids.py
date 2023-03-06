# This is a python script to upload additional data from the GRACEnet natres.csv file to an existing MySQL database.

import pandas
import pandas as pd
import numpy as np
import datetime
import psycopg2
import pandas.io.sql as sqlio
from config.config import config
from src.data.db_conn import load_db_table
import mysql.connector
import math
import sys
import random
import time
import os


class UpdateDatabase():
    def main(self):
        # Connect to mysql database
        cnx = mysql.connector.connect(user='root', password='Kh18riku!',
                                      host='127.0.0.1',
                                      database='global_n2o')
        cursor = cnx.cursor()
        '''
        query = ("SELECT * FROM LinearMeasurement")

        # Execute the query and fetch all results
        lineardf = pd.read_sql(query, cnx)

        # Order the dataframe first by RepID, then by Date
        lineardf = lineardf.sort_values(by=['RepID', 'Date'], ignore_index=True)
        # Reset the index to the current order
        lineardf = lineardf.reset_index(drop=True)
        # Set LinMeasID to be equal to the index
        lineardf['LinMeasID'] = lineardf.index + 1
        # Order dataframe by LinMeasID       
        lineardf = lineardf.sort_values(by=['LinMeasID'])
        # Replace all NaN values with None
        lineardf = lineardf.where(pd.notnull(lineardf), "NULL")
        '''
        # Query all data from the LinearMeasurement table, join to the TreatmentID from Replication table
        query = ("SELECT * FROM DailyData ORDER BY MeasurementID ASC")

        # Execute the query and fetch all results
        lineardf = pd.read_sql(query, cnx)
        cnx.close()

        UpdateDatabase.upload_gn_names(lineardf)

    def upload_gn_names(self, lineardf):
        # Get only first datapoint in each replication
        lineardf = lineardf.drop_duplicates(subset=['RepID'], keep='first')
        # Round the N2OFlux to 2 decimal places
        lineardf['N2OFlux'] = lineardf['N2OFlux'].round(3)
        lineardf['VWC'] = lineardf['VWC'].round(3)
        lineardf['SoilT'] = lineardf['SoilT'].round(3)
        # Convert Date and N2OFlux to string
        lineardf['Date'] = lineardf['Date'].astype(str)
        lineardf['N2OFlux'] = lineardf['N2OFlux'].astype(str)
        lineardf['VWC'] = lineardf['VWC'].astype(str)
        lineardf['SoilT'] = lineardf['SoilT'].astype(str)
        print(lineardf.head())
        rep_notfound_counter = 0
        ts_notfound_counter = 0
        noflux_counter = 0
        total_counter = 0
        missingdf = pd.DataFrame(columns=['Experiment', 'Treatment', 'Replication', 'Timeseries', 'Date', 'N2OFlux'])

        # Get all folders in the GRACEnet folder
        gn_experiments = os.listdir('GRACEnet')
        for experiment in gn_experiments:
            # Get all treatments in the experiment folder
            gn_treatment = os.listdir('GRACEnet/' + experiment)
            for treatment in gn_treatment:
                # Get all replications in the treatment folder
                gn_replication = os.listdir('GRACEnet/' + experiment + '/' + treatment)
                # For replication in the treatment folder, if there are multiple timeseries in the replication folder,
                # get all timeseries
                for replication in gn_replication:
                    # If 'precip.csv' is not in the replication folder, get all timeseries
                    if 'precip.csv' not in os.listdir('GRACEnet/' + experiment + '/' + treatment + '/' + replication):
                        gn_timeseries = os.listdir('GRACEnet/' + experiment + '/' + treatment + '/' + replication)
                        for timeseries in gn_timeseries:
                            total_counter += 1
                            if 'alldata-db.csv' not in os.listdir(
                                    'GRACEnet/' + experiment + '/' + treatment + '/' + replication + '/' + timeseries +
                                    '/processed') or 'flux.csv' not in os.listdir('GRACEnet/' + experiment + '/' +
                                                                                    treatment + '/' + replication + '/' +
                                                                                    timeseries):
                                noflux_counter += 1
                                missingdf = missingdf.append({'Experiment': experiment, 'Treatment': treatment,
                                                                'Replication': replication, 'Timeseries': timeseries,
                                                                'Date': 'NULL', 'N2OFlux': 'NULL'}, ignore_index=True)
                                continue
                            print(experiment)
                            print(treatment)
                            print(replication)
                            print(timeseries)
                            # Open flux.csv
                            fluxdf = pd.read_csv('GRACEnet/' + experiment + '/' + treatment + '/' + replication + '/' +
                                                 timeseries + '/processed/' + 'alldata-db.csv',
                                                 names=['id', 'Date', 'n2o_flux', 'soil_vwc', 'WFPS', 'soil_temp_c',
                                                        'AirT', 'Precip', 'NApplied', 'id2', 'NForm', '1', '2', '3',
                                                        '4', '5', '6', '7', '8']
                                                 )
                            current_date = fluxdf['Date'][0]
                            # Drop time from date
                            current_date = current_date.split(' ')[0]
                            current_flux = fluxdf['n2o_flux'][0]
                            current_vwc = fluxdf['soil_vwc'][0]
                            current_soiltemp = fluxdf['soil_temp_c'][0]
                            # Truncate current_flux to 2 decimal places
                            print(current_flux)
                            current_flux = round(current_flux, 3)
                            current_flux = str(current_flux)
                            current_vwc = round(current_vwc, 3)
                            current_soiltemp = round(current_soiltemp, 3)
                            current_vwc = str(current_vwc)
                            current_soiltemp = str(current_soiltemp)
                            print(current_date)
                            print(current_flux)
                            print(current_vwc)
                            print(current_soiltemp)
                            flag = 0
                            try:
                                # Get the LinMeasID, RepID, and TreamentID for the current date and flux
                                current_linmeasid = lineardf.loc[(lineardf['Date'] == current_date) &
                                                                 (lineardf['N2OFlux'] == current_flux) &
                                                                 (lineardf['VWC'] == current_vwc) &
                                                                 (lineardf['SoilT'] == current_soiltemp),
                                                                 'MeasurementID'].iloc[0]
                            except:
                                flag = 1
                                pass
                            if flag == 1:
                                print('No match found')
                                ts_notfound_counter += 1
                                missingdf = missingdf.append({'Experiment': experiment, 'Treatment': treatment,
                                                                'Replication': replication, 'Timeseries': timeseries,
                                                                'Date': current_date, 'N2OFlux': current_flux},
                                                                 ignore_index=True)
                                continue
                            currentdf = lineardf.loc[lineardf['MeasurementID'] == current_linmeasid]
                            # Convert currentdf to a list
                            current_list = currentdf.values.tolist()
                            print(current_list)
                            current_repid = current_list[0][0]
                            print(currentdf)
                            print(current_linmeasid)
                            print(current_repid)
                            # Update the Replication table with the replication name
                            cnx = mysql.connector.connect(user='root', password='Kh18riku!',
                                                          host='127.0.0.1',
                                                          database='global_n2o')
                            cursor = cnx.cursor()
                            query = ("UPDATE DailyData SET RepName = '" + replication + "' WHERE RepID = " +
                                     str(current_repid))
                            cursor.execute(query)
                            cnx.commit()
                            cnx.close()
                    # If not multiple timeseries, get the flux.csv
                    else:
                        total_counter += 1
                        if 'alldata-db.csv' not in os.listdir(
                                'GRACEnet/' + experiment + '/' + treatment + '/' + replication + '/processed') or \
                                'flux.csv' not in os.listdir('GRACEnet/' + experiment + '/' + treatment + '/' +
                                                             replication):
                            noflux_counter += 1
                            missingdf = missingdf.append({'Experiment': experiment, 'Treatment': treatment,
                                                            'Replication': replication, 'Timeseries': 'NULL',
                                                            'Date': 'NULL', 'N2OFlux': 'NULL'}, ignore_index=True)
                            continue
                        # Open flux.csv as df, set column titles
                        fluxdf = pd.read_csv('GRACEnet/' + experiment + '/' + treatment + '/' + replication + '/' +
                                             'processed/' + 'alldata-db.csv',
                                             names=['id', 'Date', 'n2o_flux', 'soil_vwc', 'WFPS', 'soil_temp_c',
                                                    'AirT', 'Precip', 'NApplied', 'id2', 'NForm', '1', '2', '3', '4',
                                                    '5', '6', '7', '8']
                                             )
                        current_date = fluxdf['Date'][0]
                        # Drop time from date
                        current_date = current_date.split(' ')[0]
                        current_flux = fluxdf['n2o_flux'][0]
                        current_vwc = fluxdf['soil_vwc'][0]
                        current_soiltemp = fluxdf['soil_temp_c'][0]
                        # Round current_flux to 2 decimal places
                        print(current_flux)
                        current_flux = round(current_flux, 3)
                        current_flux = str(current_flux)
                        print(current_date)
                        print(current_flux)
                        current_vwc = round(current_vwc, 3)
                        current_soiltemp = round(current_soiltemp, 3)
                        print(current_vwc)
                        print(current_soiltemp)
                        flag = 0
                        try:
                            # Get the LinMeasID, RepID, and TreamentID for the current date and flux
                            current_linmeasid = lineardf.loc[(lineardf['Date'] == current_date) &
                                                             (lineardf['N2OFlux'] == current_flux) &
                                                             (lineardf['VWC'] == current_vwc) &
                                                             (lineardf['SoilT'] == current_soiltemp),
                                                             'MeasurementID'].iloc[0]
                        except:
                            flag = 1
                            pass
                        if flag == 1:
                            print('No match found')
                            rep_notfound_counter += 1
                            missingdf = missingdf.append({'Experiment': experiment, 'Treatment': treatment,
                                                            'Replication': replication, 'Timeseries': 'NULL',
                                                            'Date': current_date, 'N2OFlux': current_flux},
                                                             ignore_index=True)
                            continue
                        currentdf = lineardf.loc[lineardf['MeasurementID'] == current_linmeasid]
                        # Convert currentdf to a list
                        current_list = currentdf.values.tolist()
                        print(current_list)
                        current_repid = current_list[0][0]
                        print(currentdf)
                        print(current_linmeasid)
                        print(current_repid)
                        # Update the Replication table with the replication name
                        cnx = mysql.connector.connect(user='root', password='Kh18riku!',
                                                      host='127.0.0.1',
                                                      database='global_n2o')
                        cursor = cnx.cursor()
                        query = ("UPDATE DailyData SET RepName = '" + replication + "' WHERE RepID = " +
                                 str(current_repid))
                        cursor.execute(query)
                        cnx.commit()
                        cnx.close()

        print("Total number of timeseries not found: " + str(ts_notfound_counter))
        print("Total number of replications not found: " + str(rep_notfound_counter))
        print("Total number of flux.csv files not found: " + str(noflux_counter))
        print("Total number of timeseries: " + str(total_counter))

        # Get the Replication data
        cnx = mysql.connector.connect(user='root', password='Kh18riku!',
                                      host='127.0.0.1',
                                      database='global_n2o')
        query = ("SELECT * FROM DailyData")
        # Execute the query and fetch all results
        repdf = pd.read_sql(query, cnx)
        cnx.close()

        # Print number of rows in Replication table where RepName is null
        print("Number of null reps:", repdf.loc[repdf['RepName'].isnull()].shape[0])

        # Save missingdf to a csv
        missingdf.to_csv('missing.csv')

    def truncate(self, number, decimals):
        """
        Returns a value truncated to a specific number of decimal places.
        """
        if not isinstance(decimals, int):
            raise TypeError("decimal places must be an integer.")
        elif decimals < 0:
            raise ValueError("decimal places has to be 0 or more.")
        elif decimals == 0:
            return math.trunc(number)

        factor = 10.0 ** decimals
        return math.trunc(number * factor) / factor


if __name__ == '__main__':
    UpdateDatabase = UpdateDatabase()
    UpdateDatabase.main()
