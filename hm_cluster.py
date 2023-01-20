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
import mysql.connector
import pandas.io.sql as sqlio

class cluster():
    def main(self):
        # Read database - PostgreSQL
        conn = mysql.connector.connect(user='rackett', password='j4FApKeQjC!2',
                                       host='mariadb-compx0.oit.utk.edu',
                                       database='rackett_fluxdb')
        df = sqlio.read_sql_query("""SELECT * FROM InterpolatedFlux """, conn)
        conn.close()

    def prep_data(self, df):
        date = df.date
        doy = (i.timetuple().tm_yday for i in date)
        print(day)



if __name__ == '__main__':
    hot_moment = cluster()
    hot_moment.main()
