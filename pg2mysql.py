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


class pg2mysql():
    def main(self):
        new_sitedf, pubsdf, allexpdf, alldatadf = pg2mysql.load_old_db()
        pg2mysql.fill_experiments(new_sitedf, pubsdf, allexpdf, alldatadf)

    def load_old_db(self):
        # Get data from old postgres databases
        params = config(config_db='database.ini')
        conn = psycopg2.connect(**params)

        allexpdf = sqlio.read_sql_query("""SELECT * FROM "ExperimentalMethods" """, conn)
        pubsdf = allexpdf.drop_duplicates(subset=['paper_id'])
        alldatadf = sqlio.read_sql_query("""SELECT * FROM "DailyPredictors" """, conn)

        # Connect to mysql database
        cnx = mysql.connector.connect(user='root', password='Kh18riku!',
                                      host='127.0.0.1',
                                      database='global_n2o')
        cursor = cnx.cursor()
        query = ("SELECT Latitude, Longitude, SiteID FROM Site ")

        # Execute the query and fetch all results
        new_sitedf = pd.read_sql(query, cnx)
        print(new_sitedf.head())

        # Get relevant columns from allexpdf
        sitedf = allexpdf[['latitude', 'longitude', 'paper_id']]
        old_sitedf = sitedf.drop_duplicates()
        return new_sitedf, pubsdf, allexpdf, alldatadf

    def fill_experiments(self, new_sitedf, pubsdf, allexpdf, alldatadf):
        new_sitedf = new_sitedf.round({'Latitude': 2, 'Longitude': 2})
        print(new_sitedf.head())

        # Create dataframe for experiments table
        expdf = pd.DataFrame(columns=['ExperimentID', 'ExperimentName', 'SiteID', 'PublicationID'])
        treatmentdf = pd.DataFrame(columns=['TreatmentID', 'TreatmentName', 'ExperimentID', 'PrimaryCrop',
                                            'SecondaryCrop', 'CoverCrop', 'Tillage', 'Management', 'FluxInstrument'])
        replicationdf = pd.DataFrame(columns=['RepID', 'RepName', 'TreatmentID',  'Sand', 'Silt', 'Clay', 'SOM', 'pH',
                                              'BulkDensity', 'Air_T_Meas', 'Precip_Meas', 'Soil_T_Meas', 'VWC_Meas',
                                              'WFPS_Meas'])
        daily_data_df = pd.DataFrame(
            columns=['RepID', 'RawID', 'Date', 'N2OFlux', 'VWC', 'WFPS', 'SoilT', 'AirT', 'Precip',
                     'NitrogenApplied', 'NitrogenForm', 'PlantedCrop', 'Harvest', 'Tillage', 'IrrigationApplied'
                     'IQRHM', 'AirTMax', 'AirTMin', 'SoilNH4', 'SoilNO3'])
        i = 0
        treatment_counter = 0
        rep_counter = 0
        # Iterate through pubsdf
        for index, row in pubsdf.iterrows():
            pub_id = row['paper_id']
            lat = float(round(row['latitude'], 2))
            long = float(round(row['longitude'], 2))

            # Query new_sitedf for longitude and latitude
            try:
                site_id = new_sitedf.loc[(new_sitedf['Latitude'] == lat) & (
                        new_sitedf['Longitude'] == long), 'SiteID'].iloc[0]

                # Get the experiment name associated with this publication
                exp_name = allexpdf.loc[allexpdf['paper_id'] == pub_id, 'gracenet_site'].iloc[0]

                # Add row to expdf
                expdf.loc[i] = [i + 1, exp_name, site_id, pub_id]

                # For the number of treatments in this experiment, add a row to treatmentdf
                # Get list of treatment_id's associated with this publication, remove duplicates
                pub_treatments = list(allexpdf.loc[allexpdf['paper_id'] == pub_id, 'treatment_id'].unique())
                print(pub_treatments)

                # Iterate through experiments
                for treatment in pub_treatments:
                    iter_treat = allexpdf.loc[allexpdf['treatment_id'] == treatment].iloc[0]
                    treatmentdf.loc[treatment_counter] = [treatment_counter + 1, i + 1,
                                                          iter_treat['gracenet_treatment'], iter_treat['crop'],
                                                          None, iter_treat['cover_crop'],
                                                          iter_treat['tillage'], iter_treat['management'],
                                                          iter_treat['n2o_instrument']]

                    # For the number of replications in this treatment, add a row to replicationdf
                    # Get list of replications associated with this treatment
                    current_treatment = row['treatment_id']
                    experiment_reps = list(allexpdf.loc[allexpdf['treatment_id'] == treatment, 'experiment_id'])
                    for rep_id in experiment_reps:
                        iter_rep = allexpdf.loc[allexpdf['experiment_id'] == rep_id].iloc[0]

                        # Check the first daily measurement data from alldatadf for this replication
                        rep_measurements = alldatadf.loc[alldatadf['experiment_id'] == rep_id]

                        try:
                            rep_iter = rep_measurements.iloc[0].to_frame().transpose()
                            # Assign boolean values to each measurement type based on whether or not there is data
                            air_t = not math.isnan(rep_iter['air_temp_c'])
                            precip = not math.isnan(rep_iter['precipitation_mm'])
                            soil_t = not math.isnan(rep_iter['soil_temp_c'])
                            vwc = not math.isnan(rep_iter['soil_vwc'])
                            wfps = not math.isnan(rep_iter['soil_wfps'])
                        except:
                            air_t = False
                            precip = False
                            soil_t = False
                            vwc = False
                            wfps = False

                        # Rearrange and add columns to rep_measurements to match columns of daily_data_df
                        rep_measurements['RepID'] = rep_counter + 1
                        rep_measurements['Harvest'] = None
                        rep_measurements['Tillage'] = None
                        rep_measurements['IrrigationApplied'] = None
                        rep_measurements = rep_measurements[['RepID', 'date', 'n2o_flux', 'soil_vwc', 'soil_wfps',
                                                             'soil_temp_c', 'air_temp_c', 'precipitation_mm',
                                                             'nitrogen_applied_kg_ha', 'nitrogen_form', 'planted_crop',
                                                             'Harvest', 'Tillage', 'IrrigationApplied', 'iqr_hm',
                                                             'air_temp_max', 'air_temp_min', 'nh4_mg_n_kg',
                                                             'no3_mg_n_kg', 'mgmt']]

                        # Add replications to replicationdf
                        replicationdf.loc[rep_counter] = [rep_counter + 1, iter_rep['gracenet_rep'], treatment_counter + 1,
                                                          iter_rep['percent_sand'], iter_rep['percent_silt'],
                                                          iter_rep['percent_clay'], iter_rep['om_percent'],
                                                          iter_rep['pH'], iter_rep['bulk_density'], air_t, precip,
                                                          soil_t, vwc, wfps]

                        # Add daily measurements to daily_data_df
                        daily_data_df = daily_data_df.append(rep_measurements, ignore_index=True)

                        rep_counter += 1
                    treatment_counter += 1

                i += 1

            except Exception as e:
                # Print exception error message
                print(e)

                expdf.loc[i] = [i + 1, "???", "???", pub_id]
                i += 1
                continue

        # Sort dataframes by their primary IDs
        expdf = expdf.sort_values(by=['ExperimentID'])
        treatmentdf = treatmentdf.sort_values(by=['TreatmentID'])
        replicationdf = replicationdf.sort_values(by=['RepID'])
        daily_data_df = daily_data_df.sort_values(by=['RepID', 'date'])

        # Save dataframes to csv
        expdf.to_csv('experiments.csv', index=False)
        treatmentdf.to_csv('treatments.csv', index=False)
        replicationdf.to_csv('replications.csv', index=False)
        daily_data_df.to_csv('daily_data.csv', index=False)

if __name__ == '__main__':
    pg2mysql = pg2mysql()
    pg2mysql.main()
