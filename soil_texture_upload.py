import pandas
import pandas as pd
import numpy as np
import datetime
import pandas.io.sql as sqlio
from config.config import config
import mysql.connector
import math
import sys
import random
import time
import os

# Read soil texture data from csv
texturedf = pd.read_csv('soil_texture_updated.csv')

# Connect to mysql database
cnx = mysql.connector.connect(user='root', password='Kh18riku!',
                              host='127.0.0.1',
                              database='global_n2o')
cursor = cnx.cursor()

# For row in soildf, update Replication table with soil sand, silt, and clay data on RepID
for index, row in texturedf.iterrows():
    RepID = row['RepID']
    sand = row['Sand']
    silt = row['Silt']
    clay = row['Clay']
    # If sand, silt, or clay is NaN, continue
    if math.isnan(sand) or math.isnan(silt) or math.isnan(clay):
        continue
    query = "UPDATE Replication SET Sand = %s, Silt = %s, Clay = %s WHERE RepID = %s"
    cursor.execute(query, (sand, silt, clay, RepID))
    cnx.commit()

# Close connection
cursor.close()

