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
from configparser import ConfigParser
from pathlib import Path
import mysql.connector
import pandas.io.sql as sqlio
import gesd
from sklearn.ensemble import IsolationForest
from pyod.models.mcd import MCD

class hot_moment():

    def main(self):
        # Connect to mysql database
        cnx = mysql.connector.connect(user='root', password='Kh18riku!',
                                      host='127.0.0.1',
                                      database='global_n2o')
        cursor = cnx.cursor()
        query = ("SELECT * FROM LinearMeasurement")

        # Execute the query and fetch all results
        df = pd.read_sql(query, cnx)
        cnx.close()
        #hot_moment.iqr_classifier(df)
        hot_moment.mcd_classifier(df)
        #hot_moment.genESD_classifier(df)
        #hot_moment.forest_classifier(df)
        #hot_moment.changepoint_classifier(df)
        #hot_moment.iqr_graph(df)
        #hot_moment.imr_classifier(df)
        #exp_num_list = np.arange(30,37)
        #for exp_num in exp_num_list:
        #    print(exp_num)
        #    hot_moment.imr_graph(df, exp_num)

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

    def genESD_classifier(self, df):
        sns.set_theme()
        experiments = pd.unique(df.experiment_id)
        for experiment in experiments:
            print(experiment)
            expdf = df.query('experiment_id == @experiment')
            exp_flux = list(expdf.n2o_flux)
            exp_len = len(exp_flux)
            outputdf = gesd.ESD_Test(exp_flux, 0.05, int(exp_len * .3))
            hmlol = outputdf['hot_moment'].values.tolist()
            hm_list = [item for sublist in hmlol for item in sublist]
            expdf['gesd_hm'] = (hm_list)

            # Exclude low flux anomalies
            median_flux = expdf['n2o_flux'].median()
            for index, row in expdf.iterrows():
                if row['n2o_flux'] <= median_flux and row['gesd_hm'] == 1:
                    expdf.loc[index, 'forest_hm'] = 0

            '''
            for row in expdf.index:
                print(row)
                current_id = expdf["id"][row]
                df.loc[df["id"] == current_id, "iqr_hm"] = expdf["gesd_hm"][row]
                print(df.loc[df["id"] == current_id, "iqr_hm"])
            

            colddf = expdf.copy()
            colddf.loc[colddf["gesd_hm"] == 1, "n2o_flux"] = np.nan
            hotdf = expdf.copy()
            hotdf.loc[hotdf["gesd_hm"] == 0, "n2o_flux"] = np.nan
            plt.scatter(hotdf['date'], hotdf['n2o_flux'], color='r', label='Hot Moment')
            plt.scatter(colddf['date'], colddf['n2o_flux'], color='b', label='Background')
            
            plt.ylabel('Daily Nitrous Oxide Emissions (g N/ha/d)')
            plt.legend()
            plt.show()
            '''

            for index, row in expdf.iterrows():
                df.loc[index, 'gesd_hm'] = row.gesd_hm



        try:
            conn = mysql.connector.connect(user='rackett', password='j4FApKeQjC!2',
                                           host='mariadb-compx0.oit.utk.edu',
                                           database='rackett_fluxdb')
            print('Python connected to PostgreSQL!')
        except:
            print("Failed to connect to database")
        cur = conn.cursor()
        for d in range(0, len(df)):
            QUERY = """ UPDATE InterpolatedFlux SET gesd_hm='%s' WHERE InterpolatedFlux.id='%s'
                    """ % (df['gesd_hm'][d], df['id'][d])
            cur.execute(QUERY)
        cur.execute('COMMIT')
        conn.close()


    def forest_classifier(self, df):
        experiments = pd.unique(df.experiment_id)
        for experiment in experiments:
            print(experiment)
            expdf = df.query('experiment_id == @experiment')
            expdf['index'] = expdf.index
            sampled_cols = ['index', 'n2o_flux']
            exp_flux = list(expdf.n2o_flux)
            model_IF = IsolationForest(n_jobs=-1)
            model_IF.fit(expdf[sampled_cols])
            anomaly = model_IF.predict(expdf[sampled_cols])
            expdf['forest_hm'] = [1 if i < 0 else 0 for i in anomaly]

            # Exclude low flux anomalies
            median_flux = expdf['n2o_flux'].median()
            print(median_flux)
            for index, row in expdf.iterrows():
                if row['n2o_flux'] <= median_flux and row['forest_hm'] == 1:
                    expdf.loc[index, 'forest_hm'] = 0

            plt.scatter(expdf['date'], expdf['n2o_flux'], color='b', linestyle='-', label='Daily Flux')

            colddf = expdf.copy()
            colddf.loc[colddf["forest_hm"] == 1, "n2o_flux"] = np.nan
            hotdf = expdf.copy()
            hotdf.loc[hotdf["forest_hm"] == 0, "n2o_flux"] = np.nan
            plt.scatter(hotdf['date'], hotdf['n2o_flux'], color='r', label='Hot Moment')
            plt.scatter(colddf['date'], colddf['n2o_flux'], color='b', label='Background')

            plt.ylabel('Daily Nitrous Oxide Emissions (g N/ha/d)')
            plt.legend()
            plt.show()

    def changepoint_classifier(self, df):
        experiments = pd.unique(df.experiment_id)
        for experiment in experiments:
            print(experiment)
            expdf = df.query('experiment_id == @experiment')
            exp_flux = expdf.n2o_flux.to_numpy()
            model = "rbf"
            algo = rpt.Pelt(model=model, min_size=2,).fit(exp_flux)
            result = algo.predict(pen=3)
            rpt.display(exp_flux, result, figsize=(10, 6))
            plt.title('Change Point Detection: Pelt Search Method')
            plt.show()



    def imr_graph(self, df, exp_num):
        #exp_num = 11
        expdf = df.query('experiment_id == @exp_num')
        fluxdf = expdf.n2o_flux.to_frame()
        expdf = df.query('experiment_id == @exp_num')
        if (fluxdf <= 1).values.any():
            shift = abs(fluxdf['n2o_flux'].min(axis=0)) + 1
        else:
            shift = 0
        fluxdf['n2o_flux'] += shift
        q75, q25 = np.percentile(fluxdf['n2o_flux'], [75, 25])
        iqr = q75 - q25
        threshold = q75 + iqr
        filtered_df = fluxdf.query('n2o_flux <= @threshold')
        # filtered_df = fluxdf.query('n2o_flux > 0')
        filtered_series = filtered_df.squeeze()
        original_data = expdf['n2o_flux']

        # Boxcox transform training data & save lambda value
        fitted_data, fitted_lambda = stats.boxcox(filtered_series)

        # IMR Calculations
        ibar = statistics.mean(fitted_data)
        i = 0
        mr_list = []
        while i < len(fitted_data):
            if i == 0:
                mr = 0
            else:
                mr = abs(fitted_data[i] - fitted_data[i - 1])
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

        colddf = df.loc[(df["experiment_id"] == exp_num)]
        colddf.loc[df["imr_hm"] == True, "n2o_flux"] = np.nan
        hotdf = df.loc[(df["experiment_id"] == exp_num)]
        hotdf.loc[df["imr_hm"] == False, "n2o_flux"] = np.nan

        plt.style.use('fivethirtyeight')
        plt.figure(figsize=(10, 8))
        plt.yscale('log')
        plt.scatter(hotdf['date'], hotdf['n2o_flux'], color='r', label='Hot Moment')
        plt.scatter(colddf['date'], colddf['n2o_flux'], color='b', label='Background')
        plt.axhline(y=imr_metrics[0], color='black', linestyle='dashed')
        plt.axhline(y=imr_metrics[2], color='yellow', linestyle=':')
        plt.axhline(y=imr_metrics[3], color='orange', linestyle=':')
        plt.axhline(y=imr_metrics[4], color='r', linestyle=':')
        # plt.plot(expdf['date'], expdf['n2o_flux'], color='gray')

        plt.legend()
        plt.show()

    def iqr_classifier(self, df):
        experiments = pd.unique(df.RepID)
        print(experiments)
        for experiment in experiments:
            expdf = df.loc[df['RepID'] == experiment]
            print(expdf)
            q75, q25 = np.percentile(expdf['N2OFlux'], [75, 25])
            iqr = q75 - q25
            threshold = q75 + 1.5 * iqr
            print(threshold)
            for row in expdf.index:
                print(row)
                if expdf["N2OFlux"][row] > threshold:
                    current_id = expdf["LinMeasID"][row]
                    df.loc[df["LinMeasID"] == current_id, "IQRHM"] = 1
                else:
                    current_id = expdf["LinMeasID"][row]
                    df.loc[df["LinMeasID"] == current_id, "IQRHM"] = 0
        try:
            # Connect to mysql database
            cnx = mysql.connector.connect(user='root', password='Kh18riku!',
                                          host='127.0.0.1',
                                          database='global_n2o')
            cur = cnx.cursor()
            print('Python connected to Database!')
        except:
            print("Failed to connect to database")

        for d in range(0, len(df)):
            QUERY = """ UPDATE LinearMeasurement SET IQRHM='%s' WHERE LinearMeasurement.LinMeasID='%s'
                    """ % (df['IQRHM'][d], df['LinMeasID'][d])
            cur.execute(QUERY)
        cur.execute('COMMIT')
        cnx.close()

    def mcd_classifier(self, df):
        rep_ids = df['RepID'].unique()
        for rep_id in rep_ids:
            flux = df[df['RepID'] == rep_id][['N2OFlux']]
            # Add column to flux dataframe counting upward from 0
            flux['index'] = np.arange(len(flux))
            print(flux.head())
            # MCD
            mcd = MCD(contamination=0.5)
            mcd.fit(flux)
            mcd_pred = mcd.predict(flux)
            mcd_scores = mcd.decision_scores_

            # Combine mcd_pred and mcd_scores into a dataframe
            mcd_df = pd.DataFrame({'MCD': mcd_pred, 'MCD_Score': mcd_scores})

            # Get the interquartile range
            q75, q25 = np.percentile(flux['N2OFlux'], [75, 25])
            iqr = q75 - q25
            halfiqr = iqr / 2
            median = np.median(flux['N2OFlux'])
            threshold = median + halfiqr

            # Apply filter to prevent outliers where flux is below threshold of 0.5 IQR
            df.loc[(df['RepID'] == rep_id) & (df['N2OFlux'] <= threshold), 'MCDHM'] = 0
            # Apply filter to prevent HM where flux is below 5
            df.loc[(df['RepID'] == rep_id) & (df['N2OFlux'] < 5), 'MCDHM'] = 0

        try:
            # Connect to mysql database
            cnx = mysql.connector.connect(user='root', password='Kh18riku!',
                                          host='127.0.0.1',
                                          database='global_n2o')
            cur = cnx.cursor()
            print('Python connected to Database!')
        except:
            print("Failed to connect to database")

        for d in range(0, len(df)):
            QUERY = """ UPDATE LinearMeasurement SET MCDHM='%s' WHERE LinearMeasurement.LinMeasID='%s'
                    """ % (df['MCDHM'][d], df['LinMeasID'][d])
            cur.execute(QUERY)
        cur.execute('COMMIT')
        cnx.close()

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
            conn = mysql.connector.connect(user='rackett', password='j4FApKeQjC!2',
                                           host='mariadb-compx0.oit.utk.edu',
                                           database='rackett_fluxdb')
            print('Python connected to PostgreSQL!')
        except:
            print("Failed to connect to database")
        cur = conn.cursor()
        for d in range(0, len(df)):
            QUERY = """ UPDATE "InterpolatedFlux" SET "imr_hm"='%s' WHERE "InterpolatedFlux"."id"='%s'
                    """ % (fulldf['imr_hm'][d], fulldf['id'][d])
            cur.execute(QUERY)
        cur.execute('COMMIT')
        return imr_metrics


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    hot_moment = hot_moment()
    hot_moment.main()

