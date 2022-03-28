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
        PROJECT_ROOT = get_project_root()
        # Read database - PostgreSQL
        df = load_db_table(config_db='database.ini', query='SELECT * FROM "DailyPredictors" LIMIT 5')
        print(df)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    hot_moment = hot_moment()
    hot_moment.main()

