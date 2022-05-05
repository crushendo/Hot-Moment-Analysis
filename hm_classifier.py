import random
import numpy
import pandas as pd
import statistics
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from config.config import config
from src.data.db_conn import load_db_table
from config.config import get_project_root
import psycopg2
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import math


class classifier():
    def main(self):
        df = classifier.get_data()
        X_train, X_test, y_imr, y_iqr_train, y_iqr_test = classifier.preprocess(df)
        print(y_iqr_test.head())
        print(type(y_iqr_test))
        print("Actual HM: " + str(list(y_iqr_train).count(True)))
        neighbors = np.arange(1, 10)
        score_plot = []
        for k in neighbors:
            print(k)
            y_hat_test = classifier.knn_predict(X_train, X_test, y_iqr_train, y_iqr_test, k, 2)
            score_plot.append(metrics.accuracy_score(y_iqr_test, y_hat_test))
        print(y_hat_test)
        plt.plot(neighbors, score_plot)
        plt.show()
        print("Predictions of HM: " + str(y_hat_test.count(True)))
        print("Actual HM: " + str(list(y_iqr_test).count(True)))
        print(accuracy_score(y_iqr_test, y_hat_test))

    def get_data(self):
        df = load_db_table(config_db='database.ini', query='SELECT * FROM "DailyPredictors"')
        return df

    def preprocess(self, df):
        X = df.drop(['id', 'experiment_id', 'date', 'iqr_hm', 'imr_hm', ], axis = 'columns')
        X = X[:1500]
        lower_columns = X.columns.values.tolist()
        for item in lower_columns:
            item.lower()
        X.columns = lower_columns
        X['mgmt'] = X['mgmt'].str.lower()
        X['nitrogen_form'] = X['nitrogen_form'].str.lower()
        y_iqr = df.iqr_hm
        print(y_iqr)
        y_iqr = y_iqr[:1500]
        y_imr = df.imr_hm

        # One Hot Encoding for categorical predictors- mgmt and nitrogen form
        ohe = OneHotEncoder(sparse=False)
        encoded = ohe.fit_transform(X[['mgmt']])
        column_name = ohe.get_feature_names(['mgmt'])
        encoder_df = pd.DataFrame(encoded, columns=column_name)
        X = X.join(encoder_df)

        encoded = ohe.fit_transform(X[['nitrogen_form']])
        column_name = ohe.get_feature_names(['nitrogen_form'])
        encoder_df = pd.DataFrame(encoded, columns=column_name)
        X = X.join(encoder_df)

        # Fix encoding for entries with multiple in same day
        mgmtentries = list(X['mgmt'].unique())
        nentries = list(X['nitrogen_form'].unique())
        entries = mgmtentries + nentries
        entries = list(filter(None, entries))
        for column in X.columns.values.tolist():
            if "," in column:
                if "mgmt" in column:
                    prefix = "mgmt_"
                elif "nitrogen_form" in column:
                    prefix = "nitrogen_form_"
                change_list = []
                for entry in entries:
                    if entry in column:
                        change_list.append(prefix + entry)
                i = 0
                for row in X[column]:
                    if row == 0:
                        i += 1
                        continue
                    for col in change_list:
                        X.loc[i,col] = 1
                    i += 1
        delete_list = list(filter(lambda a: "," in a, X.columns.values.tolist()))
        delete_list.append(['mgmt', 'nitrogen_form'])
        print(delete_list)
        for item in delete_list:
            X.drop(item, inplace=True, axis=1)

        # Test train split
        X_train, X_test, y_iqr_train, y_iqr_test = train_test_split(X, y_iqr, test_size=0.3, random_state=577)
        print(X_train.head())
        # Normalizing predictor data to avoid bias from scales/ranges in predictors
        scaler = StandardScaler()
        scaled_columns = ['soil_vwc', 'soil_wfps', 'soil_temp_c', 'air_temp_c', 'precipitation_mm',
                          'nitrogen_applied_kg_ha', 'nh4_mg_n_kg', 'no3_mg_n_kg', 'n2o_flux']
        scaled_features_test = X_test[scaled_columns]
        scaled_features_train = X_train[scaled_columns]

        scaled_features_test = scaler.fit_transform(scaled_features_test.values)
        scaled_features_train = scaler.transform(scaled_features_train.values)
        X_test[scaled_columns] = scaled_features_test
        X_train[scaled_columns] = scaled_features_train
        print(X_train.head())
        return X_train, X_test, y_imr, y_iqr_train, y_iqr_test


    def knn_predict(self, X_train, X_test, y_train, y_test, k, p):
        # Counter to help with label voting
        from collections import Counter
        # Make predictions on the test data
        # Need output of 1 prediction per test data point
        y_hat_test = []

        for index, test_point in X_test.iterrows():
            #print(" Progress: " + str(len(y_hat_test)) + "/" + str(len(y_test)))
            distances = []
            for index, train_point in X_train.iterrows():
                distance = classifier.minkowski_distance(test_point, train_point, p)
                distances.append(distance)

            # Store distances in a dataframe
            df_dists = pd.DataFrame(data=distances, columns=['dist'],
                                    index=y_train.index)

            # Sort distances, and only consider the k closest points
            df_nn = df_dists.sort_values(by=['dist'], axis=0)[:k]
            # Create counter object to track the labels of k closest neighbors
            counter = Counter(y_train[df_nn.index])

            # Get most common label of all the nearest neighbors
            prediction = counter.most_common()[0][0]

            # Append prediction to output list
            y_hat_test.append(prediction)
        return y_hat_test

    def minkowski_distance(self, a, b, p):
        # Store the number of dimensions
        dim = len(a)
        # Set initial distance to 0
        distance = 0
        # Calculate minkowski distance using parameter p
        for d in range(dim):
            add_distance = abs(a[d] - b[d]) ** p
            if math.isnan(add_distance):
                continue
            distance += add_distance
            print(distance)
        distance = distance ** (1 / p)
        return distance


if __name__ == '__main__':
    classifier = classifier()
    classifier.main()