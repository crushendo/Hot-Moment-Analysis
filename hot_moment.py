import random
import numpy
import pandas as pd
import statistics
import matplotlib.pyplot as plt
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

    def iqr_graph(self, df):
        exp_num = 6
        expdf = df.loc[(df["experiment_id"] == exp_num)]
        colddf = df.loc[(df["experiment_id"] == exp_num)]
        colddf.loc[df["iqr_hm"] == True, "n2o_flux"] = np.nan
        hotdf = df.loc[(df["experiment_id"] == exp_num)]
        hotdf.loc[df["iqr_hm"] == False, "n2o_flux"] = np.nan

        plt.scatter(hotdf['date'], hotdf['n2o_flux'], color='r', label='Background')
        plt.scatter(colddf['date'], colddf['n2o_flux'], color='b', label='Hot Moment')
        plt.plot(expdf['date'], expdf['n2o_flux'], color='gray')

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

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    hot_moment = hot_moment()
    hot_moment.main()

