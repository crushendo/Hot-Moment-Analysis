import random
import numpy
import pandas as pd
import statistics
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

class data_cleaner():

    def main(self):
        filename = "Temp.csv"
        n20filename = "N2Oflux.csv"
        fluxdf = pd.read_csv(n20filename, header=None, names=['Date', 'Data'])
        predictordf = pd.read_csv(filename, header=None, names=['Date', 'Data'])
        originaldf = predictordf
        fluxstart = list(fluxdf.loc[:,"Date"])[0]
        fluxend = list(fluxdf.loc[:, "Date"])[-1]
        fluxstart = datetime.strptime(fluxstart, "%Y/%m/%d %H:%M:%S")
        fluxstart = fluxstart.date()
        fluxstart = datetime.combine(fluxstart, datetime.min.time())
        fluxend = datetime.strptime(fluxend, "%Y/%m/%d %H:%M:%S")
        fluxend = fluxend.date()
        fluxend = datetime.combine(fluxend, datetime.min.time())
        dailydf = data_cleaner.daily_avg(fluxstart, fluxend, predictordf)
        df_reindexed, df_plotting = data_cleaner.interpolator(fluxstart, fluxend, dailydf)
        data_cleaner.write_csv(df_reindexed, filename)

        fig = plt.figure()

        for frame in [df_plotting, originaldf]:
            plt.plot(frame['Date'], frame['Data'])

        plt.xlim(fluxstart, fluxend)
        #plt.ylim(0, 10000)
        plt.yscale('log')
        plt.show()

    def daily_avg(self, fluxstart, fluxend, predictordf):
        print(predictordf)
        delta = timedelta(days=1)
        date = fluxstart
        datetime_series = pd.to_datetime(predictordf['Date'])
        predictordf['Date'] = datetime_series
        datetime_index = pd.DatetimeIndex(datetime_series.values)
        predictordf = predictordf.set_index(datetime_index)
        print(predictordf)
        daily_list = []
        while date <= fluxend:
            nextday = date + delta
            day_data = predictordf[predictordf['Date'].between(date, nextday)]
            if day_data.empty:
                date += delta
                continue
            day_data = list(day_data.loc[:, "Data"])
            day_data = [float(i) for i in day_data]
            day_average = statistics.mean(day_data)
            day_list = [date, day_average]
            daily_list.append(day_list)
            date += delta
        dailydf = pd.DataFrame(daily_list, columns=['Date', 'Data'])
        print(dailydf)
        return dailydf

    def interpolator(self, fluxstart, fluxend, dailydf):
        date = fluxstart
        delta = timedelta(days=1)

        datetime_series = pd.to_datetime(dailydf['Date'])
        dailydf['Date'] = datetime_series
        datetime_index = pd.DatetimeIndex(datetime_series.values)
        dailydf = dailydf.set_index(datetime_index)
        df_reindexed = dailydf.reindex(pd.date_range(start=dailydf.index.min(),
                                                     end=dailydf.index.max(),
                                                     freq='1D'), fill_value="NaN")
        print(df_reindexed)
        data_list = list(df_reindexed.loc[:,"Data"])
        datadf = pd.DataFrame(data_list, columns=['Data'])
        print(datadf)
        datadf["Data"] = datadf["Data"].astype(float)
        datadf = datadf.interpolate(method='values').ffill().bfill()
        print(datadf)
        df_reindexed['Data'] = list(datadf.loc[:,"Data"])
        df_plotting = df_reindexed
        df_plotting['Date'] = df_plotting.index
        df_reindexed = df_reindexed.drop('Date', 1)
        print(df_plotting)

        return df_reindexed, df_plotting

    def write_csv(self, df_reindexed, filename):
        df_reindexed.to_csv(filename[:-4] + "_cleaned.csv")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_cleaner = data_cleaner()
    data_cleaner.main()


