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

class gn_uploader():
    def main(self):
        reportdf = pd.read_csv("GRACEnet Processing Report.csv")
        uploaddf = reportdf.query("Greenlight == '*' & Uploaded != '*'")
        print(uploaddf.head())
        dbdf = pd.DataFrame(columns=["paper_id", "n2o_instrument", 'crop', 'latitude', 'longitude', 'management', 'tillage',
        'soil_type', 'irrigation', 'climate', 'preceding_crop', 'cover_crop', 'daily_n2o_flux_units', 'id', 'experiment_id',
        'nitrogen_form', 'soil_texture', 'percent_clay', 'percent_silt', 'percent_sand', 'mean_annual_temp_c', 'mean_annual_precip_mm',
        'preseason_nitrogen_kg_ha', 'preseason_nitrogen_form', 'preseason_nitrogen_date', 'pH', 'om_percent', 'treatment_id',
        'gracenet_site', 'gracenet_treatment', 'gracenet_rep'])
        for index, row in uploaddf.iterrows():
            paper_id = gn_uploader.create_pub(row)
            dbdf, experiment_id, bulk_density, treatment_id = gn_uploader.create_exp(row, paper_id, dbdf)
            gn_uploader.upload_timeseries(row, paper_id, experiment_id, bulk_density, treatment_id)
            reportdf.loc[index, "Uploaded"] = "*"
            reportdf.to_csv("GRACEnet Processing Report.csv", index=False)
        reportdf.to_csv("GRACEnet Processing Report.csv", index=False)
        dbdf.to_csv("Experimental Methods.csv", index=False)

    def create_pub(self, row):
        site = row.Site
        print(site)
        sitesdf = load_db_table(config_db='database.ini', query='SELECT * FROM "Publications"')
        sites_list = list(sitesdf.gracenet_site)
        max_id = list(sitesdf.id)[-1]
        if site in sites_list:
            paper_id = list(sitesdf.query("gracenet_site == @site").id)[0]
            return paper_id
        else:
            id = int(max_id) + 1
            overviewdf = pd.read_csv("Overview.csv")
            title = list(overviewdf.query("SiteID == @site").ExperimentName)[0]
            enddate = overviewdf.query("SiteID == @site").EndDate
            enddate = list(enddate)[0]
            enddate = enddate[-4:]
            print(enddate)
            persondf = pd.read_csv("Persons.csv")
            author = list(persondf.query("SiteID == @site & PrimaryContact == 'Yes'").LastName)[0]

            # Write to CSV
            datalist = [[id, title, enddate, author, None, True, site]]
            pubdb = pd.DataFrame(data=datalist, columns=[
                'id', 'title', 'pub_date', 'lead_author', 'citation', 'gracenet', 'gracenet_site'
            ])
            pubdb.to_csv("pub_csv.csv", index=False, header=False)

            try:
                params = config(config_db='database.ini')
                conn = psycopg2.connect(**params)
                print('Python connected to PostgreSQL!')
            except:
                print("Failed to connect to database")
            cur = conn.cursor()

            f = open("pub_csv.csv", 'r')
            cursor = conn.cursor()
            try:
                cursor.copy_from(f, 'Publications', sep=",", null="")
                conn.commit()
            except (Exception, psycopg2.DatabaseError) as error:
                print("Error: %s" % error)
                conn.rollback()
                cursor.close()
                cursor.query()
                return 1
            print("copy_from_file() done")
            cursor.close()
            '''
            nonevar = "NULL"
            QUERY = """ 
            INSERT INTO "Publications" (id, title, pub_date, lead_author, citation, gracenet, gracenet_site)
            VALUES ('%s', '%s', '%s', '%s', '%s', '%s', '%s')
            """ % (id, title, enddate, author, nonevar, True, site)

            print(QUERY)
            cur.execute(QUERY)
            cur.execute('COMMIT')
            '''
            return id

    def create_exp(self, row, paper_id, dbdf):
        rep = row.Rep
        site = row.Site
        treatment = row.Treatment
        print(site)
        print(treatment)
        print(rep)
        # Get starting crop of data
        expdf = load_db_table(config_db='database.ini', query='SELECT * FROM "ExperimentalMethods"')
        expdf.sort_values(by=['experiment_id'], inplace=True)
        fluxdf = pd.read_csv("MeasGHGFlux.csv")
        flux_repdf = fluxdf.query("ExpUnitID == @rep")
        nonevar = 'None'
        crops = flux_repdf.query("Crop != @nonevar")
        crops = list(crops.Crop)
        try:
            crop = list(crops)[0]
        except:
            crop = 'None'

        # Get latest ids from exp table
        last_id = list(expdf.id)[-1]
        current_id = last_id + 1
        last_exp_id = list(expdf.experiment_id)[-1]
        experiment_id = last_exp_id + 1
        treatment_id = list(expdf.query("gracenet_site == @site & gracenet_treatment == @treatment & "
                                        "gracenet_rep == @rep").treatment_id)
        if len(treatment_id) > 1:
            treatment_id = treatment_id[0]
        else:
            treatment_id = int(list(expdf.treatment_id)[-1]) + 1

        # Get experminental site data
        expdf = pd.read_csv("FieldSites.csv")
        expdf = expdf.query("SiteID == @site")
        latitude = list(expdf.Latitude)[0]
        longitude = list(expdf.Longitude)[0]
        map = list(expdf.MAPmm)[0]
        mat = list(expdf.MATdegC)[0]

        # Get soil information
        soilphysdf = pd.read_csv("MeasSoilPhys.csv")
        soilphysdf = soilphysdf.query("ExpUnitID == @rep")
        try:
            sand = list(soilphysdf["Sand%"])[0]
        except:
            sand = math.nan
        try:
            silt = list(soilphysdf["Silt%"])[0]
        except:
            silt = math.nan
        try:
            clay = list(soilphysdf["Clay%"])[0]
        except:
            clay = math.nan
        try:
            bulk_density = list(soilphysdf["BulkDensity"])[0]
        except:
            bulk_density = math.nan

        if math.isnan(sand):
            sand = None
        if math.isnan(silt):
            silt = None
        if math.isnan(clay):
            clay = None
        if math.isnan(bulk_density):
            bulk_density = None
        soilchemdf = pd.read_csv("MeasSoilChem.csv")
        soilchemdf = soilchemdf.query("ExpUnitID == @rep")
        som = list(soilchemdf["Organic C gC/kg"])
        if len(som) > 0:
            som = som[0]
        else:
            som = None
        if som == None or math.isnan(som):
            som = None
        else:
            organic = int(som) / 10
        ph = list(soilchemdf["pH"])
        if len(ph) > 0:
            ph = ph[0]
        else:
            ph = None

        # Get treatment info
        treatmentdf = pd.read_csv("Treatments.csv")
        treatmentdf = treatmentdf.query("TreatmentID == @treatment")
        tillage = list(treatmentdf["Tillage Descriptor"])[0]
        irrigation = list(treatmentdf["Irrigation"])[0]
        cover = list(treatmentdf["Cover Crop"])[0]
        fertilizer_type = list(treatmentdf["Fertilizer Amendment Class"])[0]
        try:
            if math.isnan(fertilizer_type):
                fertilizer_type = None
        except:
            pass
        organic = list(treatmentdf["Organic Management"])[0]
        if organic == "Yes":
            management = "Organic"
        elif fertilizer_type == "None" or fertilizer_type == None:
            management = "Zero Input"
        elif tillage == "No Till" or tillage == "None":
            management = "No-till"
            tillage = "No-till"
        else:
            management = "Conventional"

        # Get preseason nitrogen application info
        preseason_date = None
        preseason_kg = None
        preseason_form = None
        fertilizerdf = pd.read_csv("MgtAmendments.csv")
        fertilizerdf = fertilizerdf.query("ExpUnitID == @rep")
        series = row.Series
        print(series)
        # Determine whether there is previous data from this experiment in an earlier time series (pastdf)
        if math.isnan(series):
            timeseriesdf = pd.read_csv("GRACEnet/" + site + "/" + treatment + "/" + rep + "/processed/" + "alldata.csv")
            pastdf = None
        else:
            timeseriesdf = pd.read_csv("GRACEnet/" + site + "/" + treatment + "/" + rep + "/" + str(int(series)) +
                                       "/processed/" + "alldata.csv")
            pastdf = None
            if int(series) > 1:
                pastdf = pd.read_csv("GRACEnet/" + site + "/" + treatment + "/" + rep + "/" + str(int(series) - 1) +
                                     "/processed/" + "alldata.csv")
        included_fertilizerdates = timeseriesdf[~timeseriesdf['nitrogen_form'].isnull()]
        included_fertilizerdates = list(included_fertilizerdates.Date)
        all_fertilizerdates = list(fertilizerdf.Date)
        # Get first fertilizer date already included in the time series data
        try:
            first_fert_date = included_fertilizerdates[0]
            first_fert_date = datetime.strptime(first_fert_date, "%Y-%m-%d")
        except:
            first_fert_date = []
        # Get the date where the time series starts
        try:
            first_data_date = list(timeseriesdf.Date)[0]
        except:
            timeseriesdf.query()
            return

        # If there is previous data from an earlier time series...
        first_data_date = datetime.strptime(first_data_date, "%Y-%m-%d")
        if isinstance(pastdf, pd.DataFrame) and len(list(pastdf.Date)) > 0:
            last_enddate = list(pastdf.Date)[-1]
            last_enddate = datetime.strptime(last_enddate, "%Y-%m-%d")
            date_gap = first_data_date - last_enddate
            date_gap = int(date_gap.days)
            missing_range = [last_enddate + timedelta(days=idx) for idx in range(date_gap)]
            missing_range.reverse()
            for date in missing_range:
                formatted_date = datetime.strftime(date, "%Y-%m-%d")
                if formatted_date in all_fertilizerdates:
                    preseason_date = list(fertilizerdf.query("Date == @date").Date)[0]
                    preseason_kg = list(fertilizerdf.query("Date == @date").TotalNAmount)[0]
                    preseason_form = list(fertilizerdf.query("Date == @date").AmendType)[0]
                    break
        # If there is NOT previous data from an earlier time series...
        else:
            included_fertilizerdates.reverse()
            for date in all_fertilizerdates:
                formatted_date = datetime.strptime(date, "%m/%d/%Y")
                dategap = first_data_date - formatted_date
                if int(dategap.days) > 0 & int(dategap.days) < 90:
                    preseason_date = list(fertilizerdf.query("Date == @date").Date)[0]
                    preseason_kg = list(fertilizerdf.query("Date == @date").TotalNAmount)[0]
                    preseason_form = list(fertilizerdf.query("Date == @date").AmendType)[0]
                    break


        # Upload data to postgres db
        try:
            params = config(config_db='database.ini')
            conn = psycopg2.connect(**params)
            print('Python connected to PostgreSQL!')
        except:
            print("Failed to connect to database")
        cur = conn.cursor()

        # Write to CSV
        datalist = [[paper_id, "Static Chamber", crop, latitude, longitude, management, tillage, None, irrigation, None,
                     None, cover, "g N2O-N/ha/d", current_id, experiment_id, fertilizer_type, None, clay, silt, sand,
                     mat, map, preseason_kg, preseason_form, preseason_date, ph, som, treatment_id, site, treatment,
                     rep, bulk_density]]
        expdb = pd.DataFrame(data=datalist)
        expdb.to_csv("exp_csv.csv", index=False, header=False)

        try:
            params = config(config_db='database.ini')
            conn = psycopg2.connect(**params)
            print('Python connected to PostgreSQL!')
        except:
            print("Failed to connect to database")
        cur = conn.cursor()

        f = open("exp_csv.csv", 'r')
        cursor = conn.cursor()
        try:
            cursor.copy_from(f, 'ExperimentalMethods', sep=",", null="")
            conn.commit()
        except (Exception, psycopg2.DatabaseError) as error:
            print("Error: %s" % error)
            conn.rollback()
            cursor.close()
            cursor.query()
            return 1
        print("copy_from_file() done")
        cursor.close()


        QUERY = """
        INSERT INTO "ExperimentalMethods" (paper_id, n2o_instrument, crop, latitude, longitude, management, tillage,
        soil_type, irrigation, climate, preceding_crop, cover_crop, daily_n2o_flux_units, id, experiment_id, 
        nitrogen_form, soil_texture, percent_clay, percent_silt, percent_sand, mean_annual_temp_c, mean_annual_precip_mm,
        preseason_nitrogen_kg_ha, preseason_nitrogen_form, preseason_nitrogen_date, "pH", om_percent, treatment_id,
        gracenet_site, gracenet_treatment, gracenet_rep, bulk_density)
        VALUES ('%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', 
        '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s')
        """ % (paper_id, "Static Chamber", crop, latitude, longitude, management, tillage, None, irrigation, None, None,
               cover, "g N2O-N/ha/d", current_id, experiment_id, fertilizer_type, None, clay, silt, sand, mat, map,
               preseason_kg, preseason_form, preseason_date, ph, som, treatment_id, site, treatment, rep, bulk_density)
        print(QUERY)
        rowdf = pd.DataFrame([[paper_id, "Static Chamber", crop, latitude, longitude, management, tillage, None,
                              irrigation, None, None, cover, "g N2O-N/ha/d", current_id, experiment_id, fertilizer_type,
                              None, clay, silt, sand, mat, map, preseason_kg, preseason_form, preseason_date, ph, som,
                              treatment_id, site, treatment, rep, bulk_density]],
                             columns=["paper_id", "n2o_instrument", 'crop', 'latitude', 'longitude', 'management',
                                      'tillage', 'soil_type', 'irrigation', 'climate', 'preceding_crop', 'cover_crop',
                                      'daily_n2o_flux_units', 'id', 'experiment_id', 'nitrogen_form', 'soil_texture',
                                      'percent_clay', 'percent_silt', 'percent_sand', 'mean_annual_temp_c',
                                      'mean_annual_precip_mm', 'preseason_nitrogen_kg_ha', 'preseason_nitrogen_form',
                                      'preseason_nitrogen_date', 'pH', 'om_percent', 'treatment_id', 'gracenet_site',
                                      'gracenet_treatment', 'gracenet_rep', 'bulk_density'])
        joindf = [dbdf, rowdf]
        dbdf = pd.concat(joindf)
        cur.execute(QUERY)
        cur.execute('COMMIT')
        
        return dbdf, experiment_id, bulk_density, treatment_id

    def upload_timeseries(self, row, paper_id, experiment_id, bulk_density, treatment_id):
        rep = row.Rep
        site = row.Site
        treatment = row.Treatment
        try:
            series = str(int(row.Series))
        except:
            series = ""
        if series == "":
            working_dir = "GRACEnet/" + site + "/" + treatment + "/" + rep + "/processed/"
        else:
            working_dir = "GRACEnet/" + site + "/" + treatment + "/" + rep + "/" + series + "/processed/"
        datadf = pd.read_csv(working_dir + "/alldata.csv")

        # Add WFPS column using soil bulk density and VWC
        cols = datadf.columns
        datadf['soil_wfps'] = None
        if 'soil_vwc' in cols:
            if bulk_density is not None:
                b = datadf[["soil_vwc"]].apply(lambda a: a / (1 - float(bulk_density) / 2.65))
                datadf.drop("soil_wfps", axis=1, inplace=True)
                datadf["soil_wfps"] = b

        # Format dataframe to match postgres table
        datadf['experiment_id'] = experiment_id
        datadf['treatment_id'] = treatment_id
        datadf['id'] = None
        datadf['iqr_hm'] = None
        datadf['imr_hm'] = None
        datadf['nh4_mg_n_kg'] = None
        datadf['no3_mg_n_kg'] = None
        missing_cols_list = ['soil_vwc', 'soil_temp_c', 'air_temp_c', 'precipitation_mm', 'air_temp_max',
                             'air_temp_min']
        for column in missing_cols_list:
            if column in datadf.columns:
                continue
            else:
                datadf[column] = None
        # Reorganize column order
        db_columns = ['experiment_id', 'Date', 'n2o_flux', 'soil_vwc', 'soil_wfps', 'soil_temp_c', 'air_temp_c',
                      'precipitation_mm', 'nitrogen_applied_kg_ha', 'id', 'nitrogen_form', 'mgmt', 'iqr_hm',
                      'nh4_mg_n_kg', 'no3_mg_n_kg', 'imr_hm', 'planted_crop', 'air_temp_max', 'air_temp_min']
        df_columns = datadf.columns
        shared_columns = []
        for column in db_columns:
            if column in df_columns:
                shared_columns.append(column)
        print(shared_columns)
        datadf = datadf.reindex(columns=shared_columns)

        # Upload time series data to postgres
        try:
            params = config(config_db='database.ini')
            conn = psycopg2.connect(**params)
            print('Python connected to PostgreSQL!')
        except:
            print("Failed to connect to database")
        cur = conn.cursor()

        # Fill IDs with unique values
        iddf = load_db_table(config_db='database.ini', query='SELECT * FROM "DailyPredictors"')
        iddf.sort_values(by=['id'], inplace=True)
        start_id = list(iddf.id)[-1]
        start_id = int(start_id) + 1
        dblen = datadf.shape[0]
        id_list = np.arange(start_id, start_id + dblen, 1)
        print(id_list)
        datadf['id'] = id_list
        datadf.to_csv(working_dir + "/alldata-db.csv", index=False, header=False)

        f = open(working_dir + "/alldata-db.csv", 'r')
        cursor = conn.cursor()
        try:
            cursor.copy_from(f, 'DailyPredictors', sep=",", null="")
            conn.commit()
        except (Exception, psycopg2.DatabaseError) as error:
            print("Error: %s" % error)
            conn.rollback()
            cursor.close()
            cursor.query()
            return 1
        print("copy_from_file() done")
        cursor.close()

if __name__ == '__main__':
    gn_uploader = gn_uploader()
    gn_uploader.main()
