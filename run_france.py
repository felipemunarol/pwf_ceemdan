import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["AUTOGRAPH_VERBOSITY"] = "0"

import logging
logging.getLogger("tensorflow").setLevel(logging.FATAL)

import tensorflow as tf
tf.get_logger().setLevel(logging.FATAL)
tf.autograph.set_verbosity(0)

from tensorflow.keras import backend as K

import faulthandler
faulthandler.enable()

import gc
import pandas as pd

from myfunctions_france_felipe import \
    svr_model, ann_model, rf_model,lstm_model,emd_lstm,eemd_lstm, \
    ceemdan_lstm,proposed_method, proposed_method_hilbert_transform_EMD_library, proposed_method_stable_layer, proposed_method_dropout_layer, proposed_method_stable_and_dropout_layer, \
    proposed_method_with_bilstm, proposed_method_with_gru, proposed_method_with_bigru, proposed_method_with_transformer_keras, \
    proposed_method_with_patchtransformer_tf, proposed_method_with_kan, proposed_method_with_deeponet, \
    proposed_method_with_lstm_deeponet, run_cv

# Limita uso de memória CPU. Impede que o código pare.
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from tensorflow.python.client import device_lib

print("GPUs:", [x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"])

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def clear_memory():
    K.clear_session()
    gc.collect()

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

# print('svr_model')
# res = svr_model(datasets,dates,look_back,data_partition,cap)
# res = run_cv(method=svr_model, datasets=datasets, dates=dates, look_back=look_back, cap=cap, n_splits=3)

# print('ann_model')
# res = ann_model(datasets,dates,look_back,data_partition,cap)
# res = run_cv(method=ann_model, datasets=datasets, dates=dates, look_back=look_back, cap=cap, n_splits=3)
# clear_memory()

# print('rf_model')
# res = rf_model(new_data,i,look_back,data_partition,cap)

# print('lstm_model')
# res = lstm_model(new_data,i,look_back,data_partition,cap)
# clear_memory()

# print('emd_lstm')
# res = emd_lstm(new_data,i,look_back,data_partition,cap)
# clear_memory()

# print('eemd_lstm')
# res = eemd_lstm(new_data,i,look_back,data_partition,cap)
# clear_memory()

# print('ceemdan_lstm')
# res = ceemdan_lstm(new_data,i,look_back,data_partition,cap)
# clear_memory()

print('proposed_method')
res = proposed_method(datasets, look_back, data_partition, cap)
# res = run_cv(proposed_method, datasets, dates=dates, look_back=look_back, cap=cap)
clear_memory()

# print('proposed_method using hilbert transform')
# proposed_method_hilbert_transform_EMD_library(datasets, look_back, data_partition, cap)
# res = run_cv(proposed_method_hilbert_transform_EMD_library, datasets, dates=dates, look_back=look_back, cap=cap)
# clear_memory()

# print('proposed_method_stable_and_dropout_layer')
# res = run_cv(proposed_method_stable_and_dropout_layer, datasets, dates=dates, look_back=look_back, cap=cap)
# clear_memory()

# print('proposed_method_with_bilstm')
# proposed_method_with_bilstm(datasets,look_back,data_partition,cap)
# res = run_cv(proposed_method_with_bilstm, datasets, dates=dates, look_back=look_back,cap=cap)
# clear_memory()

# print('proposed_method_with_gru')
# proposed_method_with_gru(datasets, look_back, data_partition, cap)
# res = run_cv(proposed_method_with_gru, datasets, look_back=look_back,cap=cap)
# clear_memory()

# print('proposed_method_with_bigru')
# proposed_method_with_bigru(datasets, look_back, data_partition, cap)
# res = run_cv(proposed_method_with_bigru, datasets, dates=dates, look_back=look_back,cap=cap)
# clear_memory()

# print('proposed_method_with_transformer_keras')
# proposed_method_with_transformer_keras(datasets, look_back, data_partition, cap)
# res = run_cv(proposed_method_with_transformer_keras, datasets, dates=dates, look_back=look_back,cap=cap)
# clear_memory()



