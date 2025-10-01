import pandas as pd
import numpy as np
import psycopg2
from config.config import config
from src.data.db_conn import load_db_table
import math
import darts
from darts import TimeSeries
from darts.datasets import AirPassengersDataset, MonthlyMilkDataset
from darts.metrics import mape, smape
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import OneHotEncoder
from darts.utils.timeseries_generation import datetime_attribute_timeseries
import matplotlib.pyplot as plt
from darts.models import (
    RNNModel,
    TCNModel,
    TransformerModel,
    NBEATSModel,
    BlockRNNModel,
    TFTModel
)
from darts.utils.likelihood_models import GaussianLikelihood

