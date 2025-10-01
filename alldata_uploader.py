from __future__ import print_function
import pandas as pd
import statistics
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import os
import shutil
import psycopg2
from config.config import config
from src.data.db_conn import load_db_table
import math
import mysql.connector


class gn_uploader():
    def main(self):
        alldata = pd.read_csv("processed/alldata.csv")

        # Connect to database
        cnx = mysql.connector.connect(user='root', password='Kh18riku!',
                                      host='127.0.0.1',
                                      database='global_n2o')
        cursor = cnx.cursor()

        column_list = ['experiment_id', 'Date', 'n2o_flux', 'soil_vwc', 'soil_wfps', 'soil_temp_c', 'air_temp_c',
                       'precipitation_mm', 'nitrogen_applied_kg_ha', 'nitrogen_form', 'mgmt', 'nh4_mg_n_kg',
                       'no3_mg_n_kg', 'planted_crop', 'air_temp_max', 'air_temp_min']
        # If alldata does not have all the columns in column_list, add them with None values
        for column in column_list:
            if column not in alldata.columns:
                alldata[column] = None
        alldata = alldata.replace({np.nan: None})
        print(alldata.head())
        # For each row in alldata, upload the timeseries to RawMeasurement
        for index, row in alldata.iterrows():
            print(row)
            # Insert data into the database
            RepID = row['experiment_id']
            Date = row['Date']
            N2OFlux = row['n2o_flux']
            SoilVWC = row['soil_vwc']
            SoilWFPS = row['soil_wfps']
            SoilTempC = row['soil_temp_c']
            AirTempC = row['air_temp_c']
            Precipitation_mm = row['precipitation_mm']
            NitrogenApplied_kg_ha = row['nitrogen_applied_kg_ha']
            NitrogenForm = row['nitrogen_form']
            Mgmt = row['mgmt']
            NH4_mg_N_kg = row['nh4_mg_n_kg']
            NO3_mg_N_kg = row['no3_mg_n_kg']
            PlantedCrop = row['planted_crop']
            AirTempMax = row['air_temp_max']
            AirTempMin = row['air_temp_min']

            add_data = ("INSERT INTO RawMeasurement "
                        "(RepID, Date, N2OFlux, VWC, WFPS, SoilT, AirT, Precip, "
                        "NitrogenApplied, NitrogenForm, Management, SoilNH4, SoilNO3, PlantedCrop, AirTMax, "
                        "AirTMin) "
                        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)")

            # Insert data into the database
            cursor.execute(add_data, [RepID, Date, N2OFlux, SoilVWC, SoilWFPS, SoilTempC, AirTempC, Precipitation_mm,
                                      NitrogenApplied_kg_ha, NitrogenForm, Mgmt, NH4_mg_N_kg, NO3_mg_N_kg, PlantedCrop,
                                      AirTempMax, AirTempMin])

        # Commit the changes
        cnx.commit()
        cursor.close()
    # save the data to a csv file


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    gn_uploader = gn_uploader()
    gn_uploader.main()

