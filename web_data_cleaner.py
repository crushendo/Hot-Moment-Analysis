import random
import numpy
import pandas as pd
import statistics
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

class data_cleaner():

    def main(self):
        filename = "Precip.csv"
        n20filename = "all-flux-ghad.csv"
        datatype = ""
        if "flux" in filename.lower():
            datatype = "n2o_flux"
        elif "precip" in filename.lower():
            datatype = "precipitation_mm"
        elif "air" in filename.lower():
            datatype = "air_temp_c"
        elif "soil t" in filename.lower():
            datatype = "soil_temp_c"
        elif "wfps" in filename.lower():
            datatype = "soil_wfps"
        elif "smc" in filename.lower() or "vwc" in filename.lower():
            datatype = "soil_vwc"
        elif "nit" in filename.lower():
            datatype = "nitrogen_applied_kg"
        fluxdf = pd.read_csv(n20filename, header=None, names=['Date', 'Data'])
        predictordf = pd.read_csv(filename, header=None, names=['Date', datatype])
        originaldf = predictordf
        fluxstart = list(fluxdf.loc[:,"Date"])[0]
        fluxend = list(fluxdf.loc[:, "Date"])[-1]
        try:
            fluxstart = datetime.strptime(fluxstart, "%m/%d/%Y %H:%M")
            fluxend = datetime.strptime(fluxend, "%m/%d/%Y %H:%M")
        except:
            fluxstart = datetime.strptime(fluxstart, "%Y/%m/%d")
            fluxend = datetime.strptime(fluxend, "%Y/%m/%d")
        fluxstart = fluxstart.date()
        fluxstart = datetime.combine(fluxstart, datetime.min.time())
        fluxend = fluxend.date()
        fluxend = datetime.combine(fluxend, datetime.min.time())

        # Make sure predictor start and end dates match flux start and end dates
        predend = list(predictordf.loc[:, "Date"])[-1]
        predstart = list(predictordf.loc[:, "Date"])[0]
        print(predstart)
        print(predend)
        if predend != list(fluxdf.loc[:, "Date"])[-1]:
            dict = {'Date':[list(fluxdf.loc[:, "Date"])[-1]],datatype:['0']}
            df2 = pd.DataFrame(dict)
            predictordf = pd.concat([predictordf, df2], ignore_index=True)
        if predstart != list(fluxdf.loc[:, "Date"])[0]:
            dict = {'Date':[list(fluxdf.loc[:, "Date"])[0]],datatype:['0']}
            df2 = pd.DataFrame(dict)
            predictordf = pd.concat([predictordf, df2], ignore_index=True)
        print(predictordf)

        dailydf = data_cleaner.daily_avg(fluxstart, fluxend, predictordf, datatype)
        df_reindexed, df_plotting = data_cleaner.interpolator(fluxstart, fluxend, dailydf, datatype, fluxdf)
        data_cleaner.write_csv(df_reindexed, filename)

        # Plot data before and after
        fig = plt.figure()
        for frame in [df_plotting, originaldf]:
            plt.plot(frame['Date'], frame[datatype])
        #plt.xlim(fluxstart, fluxend)
        plt.ylim(bottom=-1)
        #plt.yscale('log')
        #plt.show()

    def daily_avg(self, fluxstart, fluxend, predictordf, datatype):
        delta = timedelta(days=1)
        date = fluxstart
        datetime_series = pd.to_datetime(predictordf['Date'])
        predictordf['Date'] = datetime_series
        datetime_index = pd.DatetimeIndex(datetime_series.values)
        predictordf = predictordf.set_index(datetime_index)
        print(predictordf.head())
        daily_list = []
        while date <= fluxend:
            nextday = date + delta
            nextday = nextday - timedelta(seconds=1)
            day_data = predictordf[predictordf['Date'].between(date, nextday)]
            if day_data.empty:
                date += delta
                continue
            day_data = list(day_data.loc[:, datatype])
            print(day_data)
            day_data = [float(i) for i in day_data]
            day_average = statistics.mean(day_data)
            day_list = [date, day_average]
            daily_list.append(day_list)
            date += delta
        dailydf = pd.DataFrame(daily_list, columns=['Date', datatype])
        print(dailydf.head())
        return dailydf

    def interpolator(self, fluxstart, fluxend, dailydf, datatype, fluxdf):
        date = fluxstart
        delta = timedelta(days=1)
        datetime_series = pd.to_datetime(dailydf['Date'])
        dailydf['Date'] = datetime_series
        datetime_index = pd.DatetimeIndex(datetime_series.values)
        dailydf = dailydf.set_index(datetime_index)
        if datatype == "precipitation_mm" or datatype == "nitrogen_applied_kg":
            df_reindexed = dailydf.reindex(pd.date_range(start=dailydf.index.min(),
                                                         end=dailydf.index.max(),
                                                         freq='1D'), fill_value="0")
            data_list = list(df_reindexed.loc[:, datatype])
            datadf = pd.DataFrame(data_list, columns=[datatype])
            print(datadf.head())
            datadf[datatype] = datadf[datatype].astype(float)
        else:
            df_reindexed = dailydf.reindex(pd.date_range(start=dailydf.index.min(),
                                                         end=dailydf.index.max(),
                                                         freq='1D'), fill_value="NaN")
            print(df_reindexed.head())
            data_list = list(df_reindexed.loc[:,datatype])
            datadf = pd.DataFrame(data_list, columns=[datatype])
            print(datadf.head())
            datadf[datatype] = datadf[datatype].astype(float)
            datadf = datadf.interpolate(method='values').ffill().bfill()

        df_reindexed[datatype] = list(datadf.loc[:,datatype])
        df_plotting = df_reindexed
        df_plotting['Date'] = df_plotting.index
        print(df_plotting.head())
        print(df_reindexed.head())
        df_reindexed = df_reindexed.drop('Date', 1)
        df_reindexed.index.name = 'Date'
        print(df_reindexed.head())
        return df_reindexed, df_plotting

    def write_csv(self, df_reindexed, filename):
        df_reindexed.to_csv(filename[:-4] + "_cleaned.csv")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_cleaner = data_cleaner()
    data_cleaner.main()


