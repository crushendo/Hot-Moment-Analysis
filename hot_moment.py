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

class hot_moment():

    def main(self):


    # Take in a PostgreSQL table and outputs a pandas dataframe
    def load_db_table(config_db, query):
        params = config(config_db)
        engine = psycopg2.connect(**params)
        data = pd.read_sql(query, con=engine)
        return data



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    hot_moment = hot_moment()
    hot_moment.main()

