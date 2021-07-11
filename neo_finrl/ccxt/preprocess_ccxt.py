from finrl.config import config
from finrl.preprocessing.preprocessors import FeatureEngineer
import pandas as pd
import numpy as np

def preprocess_btc(df, is_lag=False):
    # df = df.rename(columns={'time':'date'})
    # df.insert(loc=1, column='tic',value='BTC/USDT')
    # df = df[['date','tic','open','high','low','close','volume']]
    print(df.head())
    
    
    fe = FeatureEngineer(
                        use_technical_indicator=False,
                        tech_indicator_list = config.TECHNICAL_INDICATORS_LIST,
                        use_turbulence=False,
                        user_defined_feature = False)
            
    processed = fe.preprocess_data(df)
    if is_lag:
        processed = create_lag_data(processed)
        print("lag_data:",processed)
    print("processed.columns:", len(processed.columns))
    # ary = processed.values[:,range(2,len(processed.columns))]
    ary = processed.values[:]
    data_ary = ary.astype(np.float64)
    return data_ary

# def preprocess_btc(df):
#     df = df.rename(columns={'time':'date'})
#     df.insert(loc=1, column='tic',value='BTC/USDT')
#     df = df[['date','tic','open','high','low','close','volume']]
#     print(df.head())
    
    
#     fe = FeatureEngineer(
#                         use_technical_indicator=True,
#                         tech_indicator_list = config.TECHNICAL_INDICATORS_LIST,
#                         use_turbulence=False,
#                         user_defined_feature = False)
            
#     processed = fe.preprocess_data(df)
#     ary = processed.values[:,range(2,15)]
#     data_ary = ary.astype(np.float64)
#     return data_ary


def create_lag_data(df, lag=20):
    df_lagged = df.copy()
    for window in range(1, lag + 1):
        shifted = df.shift(window)
        shifted.columns = [x + "_lag_" + str(window) for x in df.columns]
        df_lagged = pd.concat((df_lagged, shifted), axis=1)
    df_lagged = df_lagged.dropna()
    return df_lagged
