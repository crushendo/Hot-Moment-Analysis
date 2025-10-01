import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats


def test_stat(input_series, iteration, inputdf):
    y = list(inputdf['y'].values.tolist())
    y = [item for sublist in y for item in sublist]
    std_dev = np.std(y)
    avg_y = np.mean(y)
    abs_val_minus_avg = abs(y - avg_y)
    max_of_deviations = max(abs_val_minus_avg)
    max_ind = int(inputdf[['y']].idxmax())
    #max_ind = np.argmax(abs_val_minus_avg)
    cal = max_of_deviations / std_dev
    return cal, max_ind


def calculate_critical_value(size, alpha, iteration):
    t_dist = stats.t.ppf(1 - alpha / (2 * size), size - 2)
    numerator = (size - 1) * np.sqrt(np.square(t_dist))
    denominator = np.sqrt(size) * np.sqrt(size - 2 + np.square(t_dist))
    critical_value = numerator / denominator
    return critical_value


def check_values(R, C, inp, max_index, iteration, output_series):
    if R > C:
        return 'yes'
    else:
        return 'no'


def ESD_Test(input_series, alpha, max_outliers):
    inputdf = pd.DataFrame(input_series, columns=[['y']])
    output_series = pd.DataFrame(input_series, columns=[['y']])
    output_series['hot_moment'] = 0
    stats = []
    critical_vals = []
    for iterations in range(1, max_outliers + 1):
        stat, max_index = test_stat(input_series, iterations, inputdf)
        critical = calculate_critical_value(len(inputdf.index), alpha, iterations)
        check = check_values(stat, critical, inputdf, max_index, iterations, output_series)
        if check == 'yes':
            output_series.loc[max_index, ['hot_moment']] = int(1)
        inputdf.drop(max_index, inplace=True)
        critical_vals.append(critical)
        stats.append(stat)
        if stat > critical:
            max_i = iterations
    df = pd.DataFrame({'i': range(1, max_outliers + 1), 'Ri': stats, 'λi': critical_vals})

    def highlight_max(x):
        if x.i == max_i:
            return ['background-color: yellow'] * 3
        else:
            return ['background-color: white'] * 3

    df.index = df.index + 1
    return output_series
