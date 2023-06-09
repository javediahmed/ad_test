import numpy as np
import pandas as pd
import scipy.stats as st

N_FEATURES = 8
N_CLASSES = 150
ALPHA = 0.9
TEST_SIZE = 10

start_date = '2020-01-01'
end_date = '2023-05-16'

date_index = pd.date_range(start_date, end_date, freq='D')
data = np.random.normal(size=(N_CLASSES, date_index.size, N_FEATURES))
data_future = np.random.normal(size=(N_CLASSES, TEST_SIZE, N_FEATURES))



dfs = {i:pd.DataFrame(j, index=date_index) for i, j in enumerate(data)}
df = dfs[4] # Test df


def forecast(series):
    return series.mean(), series.std()









