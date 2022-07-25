from __future__ import print_function
import pandas as pd
import statistics
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import os
import shutil
from src.data.db_conn import load_db_table


class data_cleaner():
    def main(self):
        site_list = os.listdir("GRACEnet/")
        for site in site_list:
            treatment_list = os.listdir("GRACEnet/" + site)
            for treatment in treatment_list:
                rep_list = os.listdir("GRACEnet/" + site + "/" + treatment)
                for rep in rep_list:
                    series_list = os.listdir("GRACEnet/" + site + "/" + treatment + "/" + rep)
                    if len(series_list) == 0:
                        continue
                    if ".csv" in series_list[0]:
                        working_dir = "GRACEnet/" + site + "/" + treatment + "/" + rep
                        print(working_dir)
                        data_cleaner.clean_data(working_dir, site, treatment, rep)
                    else:
                        for series in series_list:
                            working_dir = "GRACEnet/" + site + "/" + treatment + "/" + rep + "/" + series
                            print(working_dir)
                            data_cleaner.clean_data(working_dir, site, treatment, rep)

    def clean_data(self, working_dir, site, treatment, rep):
        # Get management practice dates
        fert_date_list, fert_form_list, fert_quant_list, plant_date_list, plant_crop_list, till_list, \
            harvest_list = data_cleaner.get_mgmt(site, treatment, rep)
        print(working_dir)
        try:
            os.mkdir(working_dir + "/processed")
        except:
            pass
        print(working_dir)
        for file in os.listdir(working_dir):
            print(file)
            if file == "processed":
                continue
            print(working_dir)
            filename = working_dir + "/" + file
            n20filename = filename
            datatype = ""
            if "flux" in filename.lower() or "n2o" in filename.lower():
                datatype = "n2o_flux"
            elif "precip" in filename.lower():
                datatype = "precipitation_mm"
            elif "temp_max" in filename.lower():
                datatype = "air_temp_max"
            elif "temp_min" in filename.lower():
                datatype = "air_temp_min"
            elif "air" in filename.lower():
                datatype = "air_temp_c"
            elif "soil t" in filename.lower():
                datatype = "soil_temp_c"
            elif "wfps" in filename.lower():
                datatype = "soil_wfps"
            elif "smc" in filename.lower() or "vwc" in filename.lower():
                datatype = "soil_vwc"
            elif "nit" in filename.lower():
                datatype = "nitrogen_applied_kg_ha"
            elif "no3" in filename.lower():
                datatype = "no3_mg_n_kg"
            elif "nh4" in filename.lower():
                datatype = "nh4_mg_n_kg"
            print(datatype)
            print(filename)
            fluxdf = pd.read_csv(n20filename, header=None, names=['Date', 'Data'])
            predictordf = pd.read_csv(filename, header=None, names=['Date', datatype])
            if predictordf["Date"].iloc[0] == "Date" or predictordf["Date"].iloc[0] == "Weather Date":
                predictordf = predictordf.iloc[1:]
            predictordf.sort_values(by='Date', inplace=True)
            predictordf = predictordf.reset_index(drop=True)
            originaldf = predictordf

            # Make sure predictor start and end dates match flux start and end dates
            predend = list(predictordf.loc[:, "Date"])[-1]
            predstart = list(predictordf.loc[:, "Date"])[0]
            print(predstart)
            print(predend)

            # Make sure data is sorted in chronological order
            predictordf['Date'] = pd.to_datetime(predictordf['Date'])
            predictordf.sort_values(by='Date')

            dailydf = data_cleaner.daily_avg(predictordf, datatype)
            df_reindexed, df_plotting = data_cleaner.interpolator(dailydf, datatype, fluxdf, fert_date_list)
            data_cleaner.write_csv(working_dir, df_reindexed, file)

        # Combine into single file
        dir_list = os.listdir(working_dir + "/processed")
        print(dir_list)
        fulldf = ''
        for filename in dir_list:
            if "alldata" in filename:
                continue
            filename = working_dir + "/processed/" + filename
            df = pd.read_csv(filename)
            print(df.head())
            if isinstance(fulldf, str):
                fulldf = df
            else:
                fulldf = fulldf.merge(df, how='inner', on=['Date'])
        print(fulldf.head())

        # Perform unit conversions
        fulldf = data_cleaner.unit_conversion(fulldf)

        # Add management and fertilization columns
        fulldf = data_cleaner.add_management(fulldf, fert_date_list, fert_form_list, fert_quant_list, plant_date_list,
                                             plant_crop_list, till_list, harvest_list)
        print(fulldf.head())

        # Reorganize column order
        db_columns = ['experiment_id', 'Date', 'n2o_flux', 'soil_vwc', 'soil_wfps', 'soil_temp_c', 'air_temp_c',
                      'precipitation_mm', 'nitrogen_applied_kg_ha', 'nitrogen_form', 'mgmt', 'nh4_mg_n_kg',
                      'no3_mg_n_kg', 'planted_crop', 'air_temp_max', 'air_temp_min']
        df_columns = fulldf.columns
        shared_columns = []
        for column in db_columns:
            if column in df_columns:
                shared_columns.append(column)
        print(shared_columns)
        fulldf = fulldf.reindex(columns=shared_columns)
        print(fulldf.head())

        # Write to CSV
        fulldf.to_csv(working_dir + "/processed/alldata.csv", index=False)

    def get_mgmt(self, site, treatment, rep):
        fertdf = pd.read_csv("GN_Master/MgtFertilization.csv").query("ExpUnitID == @rep")
        plantdf = pd.read_csv("GN_Master/MgtPlanting.csv").query("ExpUnitID == @rep")
        tilldf = pd.read_csv("GN_Master/MgtTillage.csv").query("ExpUnitID == @rep")
        harvestdf = pd.read_csv("GN_Master/MeasResidueMgnt.csv").query("ExpUnitID == @rep")
        fertdf.sort_values(by='Date', inplace=True)
        plantdf.sort_values(by='Date', inplace=True)
        tilldf.sort_values(by='Date', inplace=True)
        harvestdf.sort_values(by='Date', inplace=True)
        fert_date_list = list(fertdf.loc[:,"Date"])
        fert_form_list = list(fertdf.loc[:,"AmendType"])
        fert_quant_list =  list(fertdf.loc[:,"TotalNAmountkgN/ha"])
        plant_date_list = list(plantdf.loc[:,"Date"])
        plant_crop_list = list(plantdf.loc[:,"Crop"])
        till_list = tilldf["Date"].unique()
        harvest_list = harvestdf["Date"].unique()
        return fert_date_list, fert_form_list, fert_quant_list, plant_date_list, plant_crop_list, till_list, \
               harvest_list


    def unit_conversion(self, fulldf):
        # N2O conversion function
        try:
            if fulldf.soil_vwc[0] > 1:
                b = fulldf[["soil_vwc"]].apply(lambda a: a / 100)
                fulldf.drop("soil_vwc", axis=1, inplace=True)
                fulldf["soil_vwc"] = b
        except:
            pass
        return fulldf

    def add_management(self, fulldf, fert_date_list, fert_form_list, fert_quant_list, plant_date_list,
                       plant_crop_list, till_list, harvest_list):
        #fulldf = fulldf.assign(experiment_id=exp_id)
        fulldf = fulldf.assign(mgmt=None)
        fulldf = fulldf.assign(nitrogen_form=None)
        fulldf = fulldf.assign(nitrogen_applied_kg_ha=0)
        fulldf = fulldf.assign(planted_crop=None)
        print(fulldf.head())
        print(till_list)
        for date in till_list:
            if date[1] == "/":
                date = "0" + date
            date = datetime.strptime(date, "%m/%d/%Y")
            date = date.strftime("%Y-%m-%d")
            date = str(date)
            index_list = fulldf.query("Date == @date").index.tolist()
            try:
                fulldf.at[index_list[0], 'mgmt'] = "tillage"
            except IndexError:
                pass
        i = 0
        for date in plant_date_list:
            if date[1] == "/":
                date = "0" + date
            date = datetime.strptime(date, "%m/%d/%Y")
            date = date.strftime("%Y-%m-%d")
            date = str(date)
            print(date)
            index_list = fulldf.query("Date == @date").index.tolist()
            try:
                fulldf.at[index_list[0], 'mgmt'] = "planting"
                fulldf.at[index_list[0], 'planted_crop'] = plant_crop_list[i]
            except IndexError:
                pass
            i += 1
        i = 0
        for date in fert_date_list:
            if date[1] == "/":
                date = "0" + date
            date = datetime.strptime(date, "%m/%d/%Y")
            date = date.strftime("%Y-%m-%d")
            date = str(date)
            try:
                index_list = fulldf.query("Date == @date").index.tolist()
                fulldf.at[index_list[0], 'mgmt'] = "fertilizer"
                fulldf.at[index_list[0], 'nitrogen_applied_kg_ha'] = fert_quant_list[i]
                fulldf.at[index_list[0], 'nitrogen_form'] = fert_form_list[i]
            except IndexError:
                pass
            i += 1
        for date in harvest_list :
            if date[1] == "/":
                date = "0" + date
            date = datetime.strptime(date, "%m/%d/%Y")
            date = date.strftime("%Y-%m-%d")
            date = str(date)
            index_list = fulldf.query("Date == @date").index.tolist()
            try:
                fulldf.at[index_list[0], 'mgmt'] = "harvest"
            except IndexError:
                pass
        return fulldf

    def daily_avg(self, predictordf, datatype):
        for index, row in predictordf.iterrows():
            #if row[datatype] == "#VALUE!":
            #    predictordf = predictordf.drop(index, axis=0)
            try:
                check = float(row[datatype])
            except:
                predictordf =predictordf.drop(index, axis=0)
        fluxstart = predictordf.iloc[:1]
        fluxstart = list(fluxstart.Date)
        fluxstart = fluxstart[0]
        date = fluxstart
        iterdate = fluxstart.date()
        my_time = datetime.min.time()
        enddate = predictordf.iloc[-1:]
        enddate = list(enddate.Date)
        enddate = enddate[0].date()
        delta = timedelta(days=1)
        datetime_series = pd.to_datetime(predictordf['Date'])
        predictordf['Date'] = datetime_series
        print(predictordf)

        datetime_index = pd.DatetimeIndex(datetime_series.values)
        predictordf = predictordf.set_index(datetime_index)


        print(predictordf)
        daily_list = []
        while iterdate <= enddate:
            nextday = iterdate + delta
            my_time = datetime.min.time()
            nextday = datetime.combine(nextday, my_time)
            iterdatetime = datetime.combine(iterdate, my_time)
            nextday = nextday - timedelta(seconds=1)
            # Get all data points occuring on a single day
            day_data = predictordf[predictordf['Date'].between(iterdatetime, nextday)]
            day_df = day_data
            if day_data.empty:
                iterdate += delta
                date += delta
                continue
            # Retreive only measurement data as a list
            day_data = list(day_data.loc[:, datatype])
            day_data = [float(i) for i in day_data]



            # Find the average value of measurements on that day
            start_second = day_df.iloc[:1]
            start_second = list(start_second.Date)[0]
            end_second = day_df.iloc[-1:]
            end_second = list(end_second.Date)[0]
            # average all values
            day_average = statistics.mean(day_data)
            day_list = [iterdate, day_average]

            # Update new list of average daily values
            daily_list.append(day_list)
            date += delta
            iterdate += delta

        # Convert daily values list into dataframe
        dailydf = pd.DataFrame(daily_list, columns=['Date', datatype])
        return dailydf

    def interpolator(self, dailydf, datatype, fluxdf, fert_date_list):
        delta = timedelta(days=1)
        sample_dates = list(dailydf.loc[:, 'Date'])
        datetime_series = pd.to_datetime(dailydf['Date'])
        dailydf['Date'] = datetime_series
        datetime_index = pd.DatetimeIndex(datetime_series.values)
        dailydf = dailydf.set_index(datetime_index)
        if datatype == "precipitation_mm" or datatype == "nitrogen_applied_kg":
            print(dailydf)
            df_reindexed = dailydf.reindex(pd.date_range(start=dailydf.index.min(),
                                                         end=dailydf.index.max(),
                                                         freq='1D'), fill_value="0")
            data_list = list(df_reindexed.loc[:, datatype])
            datadf = pd.DataFrame(data_list, columns=[datatype])
            datadf[datatype] = datadf[datatype].astype(float)
        elif datatype == "nh4_mg_n_kg" or datatype == "no3_mg_n_kg":
            # for nitrogen sample data, if fertilizer has been applied between two data points, interpolate
            # as a stepwise function between those datapoints around the fertilization date
            df_reindexed = dailydf.reindex(pd.date_range(start=dailydf.index.min(),
                                                         end=dailydf.index.max(),
                                                         freq='1D'), fill_value="NaN")
            data_list = list(df_reindexed.loc[:, datatype])
            datadf = pd.DataFrame(data_list, columns=[datatype])
            datadf[datatype] = datadf[datatype].astype(float)
            for i in np.arange(0, len(sample_dates) - 1):
                n_date = sample_dates[i]
                np1_date = sample_dates[i+1]
                for date in fert_date_list:
                    date = datetime.strptime(date, "%m/%d/%y")
                    date= date.date()
                    if date > n_date and date < np1_date:
                        print("here")
                        low_value = df_reindexed.query("Date == @n_date")[datatype].tolist()[0]
                        high_value = df_reindexed.query("Date == @np1_date")[datatype].tolist()[0]
                        df_reindexed.loc[n_date:date, datatype] = low_value
                        df_reindexed.loc[date:np1_date, datatype] = high_value

            data_list = list(df_reindexed.loc[:, datatype])
            datadf = pd.DataFrame(data_list, columns=[datatype])
            datadf[datatype] = datadf[datatype].astype(float)
            datadf = datadf.interpolate(method='values').ffill().bfill()
        else:
            df_reindexed = dailydf.reindex(pd.date_range(start=dailydf.index.min(),
                                                         end=dailydf.index.max(),
                                                         freq='1D'), fill_value="NaN")
            data_list = list(df_reindexed.loc[:, datatype])
            datadf = pd.DataFrame(data_list, columns=[datatype])
            datadf[datatype] = datadf[datatype].astype(float)
            datadf = datadf.interpolate(method='values').ffill().bfill()

        df_reindexed[datatype] = list(datadf.loc[:,datatype])
        df_plotting = df_reindexed
        df_plotting['Date'] = df_plotting.index
        df_reindexed = df_reindexed.drop('Date', 1)
        df_reindexed.index.name = 'Date'
        return df_reindexed, df_plotting

    def write_csv(self, working_dir, df_reindexed, file):
        print(file)
        print(file[:-4])
        df_reindexed.to_csv(working_dir + "/processed/" + file[:-4] + "_cleaned.csv")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_cleaner = data_cleaner()
    data_cleaner.main()


