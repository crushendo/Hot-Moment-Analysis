import random
import numpy
import pandas as pd
import statistics

import scipy.special
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import psycopg2
from configparser import ConfigParser
from pathlib import Path
from config.config import config
from src.data.db_conn import load_db_table
from config.config import get_project_root

class hot_moment():

    def main(self):
        # Project root
        PROJECT_ROOT = get_project_root()  # Read database - PostgreSQL
        df = load_db_table(config_db='database.ini', query='SELECT * FROM "DailyPredictors"')
        print(df)
        #hot_moment.iqr_classifier(df)
        df = load_db_table(config_db='database.ini', query='SELECT * FROM "DailyPredictors"')
        hot_moment.iqr_graph(df)
        #hot_moment.imr_classifier(df)

    def iqr_graph(self, df):
        exp_num = 33
        expdf = df.loc[(df["experiment_id"] == exp_num)]
        colddf = df.loc[(df["experiment_id"] == exp_num)]
        colddf.loc[df["iqr_hm"] == True, "n2o_flux"] = np.nan
        hotdf = df.loc[(df["experiment_id"] == exp_num)]
        hotdf.loc[df["iqr_hm"] == False, "n2o_flux"] = np.nan

        plt.scatter(hotdf['date'], hotdf['n2o_flux'], color='r', label='Hot Moment')
        plt.scatter(colddf['date'], colddf['n2o_flux'], color='b', label='Background')
        #plt.plot(expdf['date'], expdf['n2o_flux'], color='gray')

        plt.legend()
        plt.show()

    def iqr_classifier(self, df):
        experiments = pd.unique(df.experiment_id)
        print(experiments)
        for experiment in experiments:
            expdf = df.loc[df['experiment_id'] == experiment]
            print(expdf)
            q75, q25 = np.percentile(expdf['n2o_flux'], [75, 25])
            iqr = q75 - q25
            threshold = q75 + iqr
            print(threshold)
            for row in expdf.index:
                print(row)
                if expdf["n2o_flux"][row] > threshold:
                    current_id = expdf["id"][row]
                    df.loc[df["id"] == current_id, "iqr_hm"] = 1
                else:
                    current_id = expdf["id"][row]
                    df.loc[df["id"] == current_id, "iqr_hm"] = 0
        try:
            params = config(config_db='database.ini')
            conn = psycopg2.connect(**params)
            print('Python connected to PostgreSQL!')
        except:
            print("Failed to connect to database")
        cur = conn.cursor()
        for d in range(0, len(df)):
            QUERY = """ UPDATE "DailyPredictors" SET "iqr_hm"='%s' WHERE "DailyPredictors"."id"='%s'
                    """ % (df['iqr_hm'][d], df['id'][d])
            cur.execute(QUERY)
        cur.execute('COMMIT')

    def imr_classifier(self, df):
        fulldf = df
        experiments = pd.unique(df.experiment_id)
        print(experiments)
        for experiment in experiments:
            print(experiment)
            expdf = df.query('experiment_id == @experiment')
            fluxdf = expdf.n2o_flux.to_frame()
            expdf = df.query('experiment_id == @experiment')
            if (fluxdf <= 1).values.any():
                shift = abs(fluxdf['n2o_flux'].min(axis = 0)) + 1
            else:
                shift = 0
            fluxdf['n2o_flux'] += shift
            q75, q25 = np.percentile(fluxdf['n2o_flux'], [75, 25])
            iqr = q75 - q25
            threshold = q75 + iqr
            filtered_df = fluxdf.query('n2o_flux <= @threshold')
            #filtered_df = fluxdf.query('n2o_flux > 0')
            filtered_series = filtered_df.squeeze()
            original_data = expdf['n2o_flux']

            # Boxcox transform training data & save lambda value
            fitted_data, fitted_lambda = stats.boxcox(filtered_series)

            # IMR Calculations
            ibar = statistics.mean(fitted_data)
            i=0
            mr_list = []
            while i < len(fitted_data):
                if i==0:
                    mr = 0
                else:
                    mr = abs(fitted_data[i] - fitted_data[i-1])
                mr_list.append(mr)
                i += 1
            mr_bar = statistics.mean(mr_list)
            i_sigma = mr_bar / 1.128
            pos_1sig = ibar + i_sigma
            pos_2sig = ibar + 2 * i_sigma
            pos_3sig = ibar + 3 * i_sigma
            neg_1sig = ibar - i_sigma
            neg_2sig = ibar - 2 * i_sigma
            neg_3sig = ibar - 3 * i_sigma
            imr_metrics = [ibar, mr_bar, pos_1sig, pos_2sig, pos_3sig, neg_1sig, neg_2sig, neg_3sig]
            print(imr_metrics)
            # Transform IMR metrics back to original scale
            imr_metrics = scipy.special.inv_boxcox(imr_metrics, fitted_lambda)
            imr_metrics = [x - shift for x in imr_metrics]
            print(imr_metrics)
            pos_3sig = imr_metrics[4]
            '''
            # Graph original data with IMR bars
            plt.scatter(expdf['date'], expdf['n2o_flux'], color='b', label='Flux')
            plt.axhline(y=irm_metrics[0], color='r', linestyle='dashed')
            plt.axhline(y=irm_metrics[2], color='r', linestyle=':')
            plt.axhline(y=irm_metrics[3], color='r', linestyle=':')
            plt.axhline(y=irm_metrics[4], color='r', linestyle=':')
            plt.yscale('log')
            plt.show()
            '''
            for row in expdf.index:
                print(row)
                if expdf["n2o_flux"][row] > pos_3sig:
                    current_id = expdf["id"][row]
                    fulldf.loc[fulldf["id"] == current_id, "imr_hm"] = 1
                else:
                    current_id = expdf["id"][row]
                    fulldf.loc[fulldf["id"] == current_id, "imr_hm"] = 0
        print(fulldf)
        try:
            params = config(config_db='database.ini')
            conn = psycopg2.connect(**params)
            print('Python connected to PostgreSQL!')
        except:
            print("Failed to connect to database")
        cur = conn.cursor()
        for d in range(0, len(df)):
            QUERY = """ UPDATE "DailyPredictors" SET "imr_hm"='%s' WHERE "DailyPredictors"."id"='%s'
                    """ % (fulldf['imr_hm'][d], fulldf['id'][d])
            cur.execute(QUERY)
        cur.execute('COMMIT')



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    hot_moment = hot_moment()
    hot_moment.main()

