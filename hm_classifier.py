import random
import numpy
import pandas as pd
import statistics
import matplotlib.pyplot as plt
import numpy as np
from src.data.db_conn import load_db_table
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import math
import plotly.graph_objects as go
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib.colors import ListedColormap


class classifier():
    def main(self):
        df = classifier.get_data()
        df2 = df.loc[:]
        X, X_train, X_test, y_imr, y_iqr_train, y_iqr_test = classifier.preprocess(df2)
        #neighbors, score_plot_1 = classifier.tune_model(X, X_train, X_test, y_imr, y_iqr_train, y_iqr_test)
        classifier.partial_model(X_train, X_test, y_imr, y_iqr_train, y_iqr_test)

    def partial_model(self, X_train, X_test, y_imr, y_iqr_train, y_iqr_test):
        num_rows = X_train.shape[0]
        print('Number of Rows in DataFrame :', num_rows)
        X_train.drop(['mgmt_None', 'nitrogen_form_None'], axis='columns', inplace=True)
        X_test.drop(['mgmt_None', 'nitrogen_form_None'], axis='columns', inplace=True)
        X = X_train.append(X_test)
        y_iqr = y_iqr_train.append(y_iqr_test)
        y_iqr = y_iqr.astype(int)
        y_iqr_train = y_iqr_train.astype(int)
        y_iqr_test = y_iqr_test.astype(int)
        print(y_iqr)
        print(X_train.isnull().any())
        neighbors = np.arange(2, 3)
        score_plot = []
        for i in neighbors:
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(X_train, y_iqr_train)
            knn_pred = knn.predict(X_test)
            score_plot.append(metrics.accuracy_score(y_iqr_test, knn_pred))
        plt.style.use('fivethirtyeight')
        plt.figure(figsize=(10, 8))
        plt.plot(neighbors, score_plot)
        plt.xlabel('Neighbors')
        plt.ylabel('Model Accuracy')
        plt.legend()
        #plt.show()
        plt.clf()

        # roc curve for models
        knn_prob = knn.predict_proba(X_test)
        fpr1, tpr1, thresh1 = roc_curve(y_iqr_test, knn_prob[:, 1], pos_label=1)
        random_probs = [0 for i in range(len(y_iqr_test))]
        p_fpr, p_tpr, _ = roc_curve(y_iqr_test, random_probs, pos_label=1)
        auc_score = roc_auc_score(y_iqr_test, knn_prob[:, 1])
        print("AUC Curve", auc_score)

        # plot roc curves
        plt.style.use('seaborn')
        plt.plot(fpr1, tpr1, linestyle='--', color='orange', label='KNN')
        plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive rate')
        plt.legend(loc='best')
        #plt.show()
        plt.clf()

        # Plot Decision Boundary
        mesh_size = .02
        margin = 0.25
        print(X)
        # Create a mesh grid on which we will run our model
        '''
        x_min, x_max = X.iloc[:, 0].min() - margin, X.iloc[:, 0].max() + margin
        y_min, y_max = X.iloc[:, 1].min() - margin, X.iloc[:, 1].max() + margin
        xrange = np.arange(x_min, x_max, mesh_size)
        yrange = np.arange(y_min, y_max, mesh_size)
        xx, yy = np.meshgrid(xrange, yrange)
        knn = KNeighborsClassifier(2, weights='uniform')
        knn.fit(X.iloc[:, 0:2], y_iqr)
        Z = knn.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)
        X = X.iloc[:, 0:2]
        print(y_iqr_train)
        trace_specs = [
            [X_train, y_iqr_train, '0', 'Train', 'square'],
            [X_train, y_iqr_train, '1', 'Train', 'circle'],
            [X_test, y_iqr_test, '0', 'Test', 'square-dot'],
            [X_test, y_iqr_test, '1', 'Test', 'circle-dot']
        ]

        fig = go.Figure(data=[
            go.Scatter(
                x=X[y == label, 0], y=X[y == label, 1],
                name=f'{split} Split, Label {label}',
                mode='markers', marker_symbol=marker
            )
            for X, y, label, split, marker in trace_specs
        ])
        fig.update_traces(
            marker_size=12, marker_line_width=1.5,
            marker_color="lightyellow"
        )

        fig.add_trace(
            go.Contour(
                x=xrange,
                y=yrange,
                z=Z,
                showscale=False,
                colorscale='RdBu',
                opacity=0.4,
                name='Score',
                hoverinfo='skip'
            )
        )
        fig.show()
        fig.write_image("fig1.png")
        '''
        knn = KNeighborsClassifier(2, weights='uniform')
        knn.fit(X.iloc[:, [0,6]], y_iqr)
        X = X.iloc[:, [0,6]].values
        h = .02

        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
        plt.xlabel('Soil WFPS')

        plt.ylabel('Soil Nitrate Concentration')
        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y_iqr, cmap=cmap_bold)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("KNN Decision Boundary")
        plt.show()


    def decision_boundary(self, X, y, X_train, X_test, y_train, y_test):


        # Create classifier, run predictions on grid
        clf = KNeighborsClassifier(2, weights='uniform')
        clf.fit(X, y)


    def get_data(self):
        df = load_db_table(config_db='database.ini', query='SELECT * FROM "DailyPredictors" WHERE "experiment_id" > 25 '
                                                           'AND "experiment_id" < 31')
        return df

    def tune_model(self, X, X_train, X_test, y_imr, y_iqr_train, y_iqr_test):
        neighbors = np.arange(7, 8, 1)
        score_plot_1 = []
        for k in neighbors:
            print("Neighbor: " + str(k))
            y_hat_test_1 = classifier.knn_predict(X_train, X_test, y_iqr_train, y_iqr_test, k, 2)
            score_plot_1.append(metrics.accuracy_score(y_iqr_test, y_hat_test_1))
        plt.style.use('fivethirtyeight')
        plt.figure(figsize=(10, 8))
        plt.plot(neighbors, score_plot_1, label='Accuracy')
        plt.xlabel('Neighbors')
        plt.ylabel('Model Accuracy')
        plt.legend()
        plt.savefig('knn_tuning_full_imr.png')
        plt.show()
        return neighbors, score_plot_1

    def preprocess(self, df):
        df.dropna(subset=['air_temp_c', 'nh4_mg_n_kg', 'no3_mg_n_kg', 'soil_wfps', 'soil_temp_c',], inplace=True)
        df["nitrogen_applied_kg_ha"].fillna(0, inplace=True)
        X = df.drop(['id', 'experiment_id', 'date', 'iqr_hm', 'imr_hm', 'n2o_flux', 'soil_vwc',
                     ], axis = 'columns')
        lower_columns = X.columns.values.tolist()
        for item in lower_columns:
            item.lower()
        X.columns = lower_columns
        X['mgmt'] = X['mgmt'].str.lower()
        X['nitrogen_form'] = X['nitrogen_form'].str.lower()
        y_iqr = df.iqr_hm
        y_iqr = y_iqr
        y_imr = df.imr_hm
        #y_iqr = y_imr

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
        for item in delete_list:
            X.drop(item, inplace=True, axis=1)
        # Test train split
        X_train, X_test, y_iqr_train, y_iqr_test = train_test_split(X, y_iqr, test_size=0.3, random_state=577)
        # Normalizing predictor data to avoid bias from scales/ranges in predictors
        scaler = StandardScaler()
        scaled_columns = ['soil_wfps', 'soil_temp_c', 'air_temp_c', 'precipitation_mm',
                          'nitrogen_applied_kg_ha', 'nh4_mg_n_kg', 'no3_mg_n_kg']
        scaled_features_test = X_test[scaled_columns]
        scaled_features_train = X_train[scaled_columns]

        scaled_features_test = scaler.fit_transform(scaled_features_test.values)
        scaled_features_train = scaler.transform(scaled_features_train.values)
        X_test[scaled_columns] = scaled_features_test
        X_train[scaled_columns] = scaled_features_train
        return X, X_train, X_test, y_imr, y_iqr_train, y_iqr_test

    def knn_predict(self, X_train, X_test, y_train, y_test, k, p):
        # Counter to help with label voting
        from collections import Counter
        # Make predictions on the test data
        # Need output of 1 prediction per test data point
        y_hat_test = []

        for index, test_point in X_test.iterrows():
            print(" Progress: " + str(len(y_hat_test)) + "/" + str(len(y_test)))
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
        distance = distance ** (1 / p)
        return distance


if __name__ == '__main__':
    classifier = classifier()
    classifier.main()