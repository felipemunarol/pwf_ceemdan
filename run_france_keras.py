import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["AUTOGRAPH_VERBOSITY"] = "0"

import faulthandler
faulthandler.enable()

import pandas as pd

import tensorflow as tf

from myfunctions_france_keras import \
    run_cv, svr_model, ann_model, rf_model, lstm_model, emd_lstm, eemd_lstm, \
    ceemdan_lstm, proposed_method 

from logger_config import configurar_logger

# Reconfigura o logger após os imports para garantir que permaneça habilitado.
logger = configurar_logger("ceemdan_keras")
logger.disabled = False

# Limita uso de memória CPU. Impede que o código pare.
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from tensorflow.python.client import device_lib

logger.info(f"GPUs: {[x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def log_model_start(name):
    logger.info("\n" + "=" * 70)
    logger.info("INICIANDO MODELO: %s", name.upper())
    logger.info("=" * 70)


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

log_model_start('svr_model')
# res = svr_model(datasets,dates,look_back,data_partition,cap)
res = run_cv(method=svr_model, datasets=datasets, dates=dates, look_back=look_back, cap=cap, n_splits=3, flag='svr')

log_model_start('ann_model')
# res = ann_model(datasets,dates,look_back,data_partition,cap)
res = run_cv(method=ann_model, datasets=datasets, dates=dates, look_back=look_back, cap=cap, n_splits=3, flag='ann')

# log_model_start('rf_model')
# res = rf_model(new_data,i,look_back,data_partition,cap)

# log_model_start('lstm_model')
# res = lstm_model(new_data,i,look_back,data_partition,cap)

# log_model_start('emd_lstm')
# res = emd_lstm(new_data,i,look_back,data_partition,cap)

# log_model_start('eemd_lstm')
# res = eemd_lstm(new_data,i,look_back,data_partition,cap)

# log_model_start('ceemdan_lstm')
# res = ceemdan_lstm(new_data,i,look_back,data_partition,cap)

log_model_start('proposed_method')
# res = proposed_method(datasets, look_back, data_partition, cap)
res = run_cv(proposed_method, datasets, dates=dates, look_back=look_back, cap=cap, lstm=True, flag='proposed_method')

log_model_start('proposed_method_stable_and_dropout_layer')
# proposed_method(datasets, look_back, data_partition, cap, drop_extend=True)
res = run_cv(proposed_method, datasets, dates=dates,
             look_back=look_back, cap=cap, drop_extend=True, flag='proposed_method_stable_and_dropout')

log_model_start('proposed_method_with_bilstm')
# # proposed_method(datasets, look_back, data_partition, cap, lstm=False, bilstm=True)
res = run_cv(proposed_method, datasets, dates=dates,
             look_back=look_back,cap=cap, lstm=False, bilstm=True, flag='bilstm')

log_model_start('proposed_method_with_gru')
# # proposed_method(datasets, look_back, data_partition, cap, lstm=False, gru=True)
res = run_cv(proposed_method, datasets, dates=dates,
             look_back=look_back,cap=cap, lstm=False, gru=True, flag='gru')

log_model_start('proposed_method_with_bigru')
# # proposed_method(datasets, look_back, data_partition, cap, lstm=False, bigru=True)
res = run_cv(proposed_method, datasets, dates=dates,
             look_back=look_back,cap=cap, lstm=False, bigru=True, flag='bigru')

log_model_start('proposed_method_with_transformer_keras')
# proposed_method(datasets, look_back, data_partition, cap, lstm=False, transformer=True)
res = run_cv(proposed_method, datasets, dates=dates,
             look_back=look_back,cap=cap, lstm=False, transformer=True, flag='transformer')

log_model_start('proposed_method using hilbert transform')
# proposed_method(datasets, look_back, data_partition, cap, hilbert=True)
res = run_cv(proposed_method, datasets, dates=dates, 
             look_back=look_back, cap=cap, hilbert=True, flag='hilbert')


