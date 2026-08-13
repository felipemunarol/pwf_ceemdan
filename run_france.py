# %% [markdown]
# **Load Data**

# %%
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tensorflow as tf
import gc
from tensorflow.keras import backend as K

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

# %%
import pandas as pd
df = pd.read_csv('dataset/final_la_haute_R0711.csv')
df['Date'] = pd.to_datetime(df['Date_time'], format='%m/%d/%Y %H:%M', errors='coerce')
df['Year'] = df['Date'].dt.year 
df['Month'] = df['Date'].dt.month 
new_data=df[['Month','Year','Date','P_avg']]
new_data=new_data[new_data.Year == 2017]

cap=max(new_data['P_avg'])


# %% [markdown]
# **Parameter Settings**

# %%
i=[1] # enter month value, i.e January = 1
look_back=6
data_partition=0.8

# %%
from myfunctions_france_felipe import \
    svr_model, ann_model, rf_model,lstm_model,emd_lstm,eemd_lstm, \
    ceemdan_lstm,proposed_method, proposed_method_hilbert_transform_EMD_library, proposed_method_stable_layer, proposed_method_dropout_layer, proposed_method_stable_and_dropout_layer, \
    proposed_method_with_bilstm, proposed_method_with_gru, proposed_method_with_bigru, proposed_method_with_transformer_keras, \
    proposed_method_with_patchtransformer_tf, proposed_method_with_kan, proposed_method_with_deeponet, \
    proposed_method_with_lstm_deeponet, run_cv_hybrid_models


# print('svr_model')
# res = svr_model(new_data,i,look_back,data_partition,cap,use_cv=True)

# print('ann_model')
# res = ann_model(new_data,i,look_back,data_partition,cap,use_cv=True)
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

# print('proposed_method')
# res = proposed_method(new_data,i,look_back,data_partition,cap)
# print('proposed_method_cv')
# res = run_cv_hybrid_models(proposed_method, new_data, i, look_back=look_back, cap=cap)
# clear_memory()

# print('proposed_method using hilbert transform')
# proposed_method_hilbert_transform_EMD_library(new_data,i,look_back,data_partition,cap)
# print('proposed_method using hilbert transform cv')
# res = run_cv_hybrid_models(proposed_method_hilbert_transform_EMD_library, new_data, i, look_back=look_back,cap=cap)
# clear_memory()

# print('proposed_method_stable_layer')
# proposed_method_stable_layer(new_data,i,look_back,data_partition,cap)
# print('proposed_method_stable_layer_cv')
# res = run_cv_hybrid_models(proposed_method_stable_layer, new_data, i, look_back=look_back,cap=cap)
# clear_memory()

# print('proposed_method_dropout_layer')
# proposed_method_dropout_layer(new_data,i,look_back,data_partition,cap)
# print('proposed_method_dropout_layer_cv')
# res = run_cv_hybrid_models(proposed_method_dropout_layer, new_data, i, look_back=look_back,cap=cap)
# clear_memory()

# print('proposed_method_stable_and_dropout_layer')
# proposed_method_stable_and_dropout_layer(new_data,i,look_back,data_partition,cap)
print('proposed_method_stable_and_dropout_layer_cv')
res = run_cv_hybrid_models(proposed_method_stable_and_dropout_layer, new_data, i, look_back=look_back,cap=cap)
clear_memory()

# print('proposed_method_with_bilstm')
# proposed_method_with_bilstm(new_data,i,look_back,data_partition,cap)
# print('proposed_method_with_bilstm_cv')
# res = run_cv_hybrid_models(proposed_method_with_bilstm, new_data, i, look_back=look_back,cap=cap)
# clear_memory()

# print('proposed_method_with_gru')
# proposed_method_with_gru(new_data,i,look_back,data_partition,cap)
# print('proposed_method_with_gru_cv')
# res = run_cv_hybrid_models(proposed_method_with_gru, new_data, i, look_back=look_back,cap=cap)
# clear_memory()

# print('proposed_method_with_bigru')
# proposed_method_with_bigru(new_data,i,look_back,data_partition,cap)
# print('proposed_method_with_bigru_cv')
# res = run_cv_hybrid_models(proposed_method_with_bigru, new_data, i, look_back=look_back,cap=cap)
# clear_memory()

# print('proposed_method_with_transformer_keras')
# proposed_method_with_transformer_keras(new_data,i,look_back,data_partition,cap)
# print('proposed_method_with_transformer_keras_cv')
# res = run_cv_hybrid_models(proposed_method_with_transformer_keras, new_data, i, look_back=look_back,cap=cap)
# clear_memory()



