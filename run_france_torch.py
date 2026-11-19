import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["AUTOGRAPH_VERBOSITY"] = "0"

import faulthandler
faulthandler.enable()

import pandas as pd

from myfunctions_france_torch import proposed_method, \
    proposed_method_with_deeponet, \
    proposed_method_with_lstm_deeponet, run_cv

from logger_config import configurar_logger

# Reconfigura o logger após os imports para garantir que permaneça habilitado.
logger = configurar_logger("ceemdan_torch")
logger.disabled = False

print("Após imports:", logger.disabled)
logger.info("Logger funcionando após imports")

df = pd.read_csv('dataset/final_la_haute_R0711.csv')
df['Date'] = pd.to_datetime(
    df['Date_time'].str.strip(),
    format='%m/%d/%Y %H:%M',
    errors='coerce'
)
df['Year'] = df['Date'].dt.year 
df['Month'] = df['Date'].dt.month 
new_data=df[['Month','Year','Date','P_avg']]
new_data=new_data[new_data.Year == 2017]
cap=max(new_data['P_avg'])

month=[1] # enter month value, i.e January = 1
look_back=6
data_partition=0.8

x = month
data1 = new_data.loc[new_data['Month'].isin(x)]
data1 = data1.reset_index(drop=True).dropna()
dates = data1['Date']
datasets = data1['P_avg'].values.reshape(-1, 1)

import logging
print("Logger disabled:", logger.disabled)
print("INFO habilitado:", logger.isEnabledFor(logging.INFO))

logger.info("TESTE INFO")
logger.warning("TESTE WARNING")

logger.info('proposed_method_with_seq_kan')
# proposed_method(datasets, look_back, data_partition, cap)
res = run_cv(proposed_method, datasets, dates=dates,
             look_back=look_back,cap=cap, flag='seqkan')




