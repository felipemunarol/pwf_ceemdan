#!/usr/bin/env python
# coding: utf-8

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["AUTOGRAPH_VERBOSITY"] = "0"
os.environ["PYTHONHASHSEED"] = "1234"

import logging

logger = logging.getLogger("ceemdan_keras")

from PyEMD import EMD, EEMD, CEEMDAN
from scipy.signal import hilbert
import ewtpy
import math
import numpy
import gc
import emd

import warnings
import logging
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("tensorflow").setLevel(logging.FATAL)

import tensorflow as tf
tf.get_logger().setLevel("FATAL")
tf.autograph.set_verbosity(0)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU
from keras.layers import LSTM, Bidirectional
from keras import backend as K
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
import numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import tensorflow_docs as tfdocs

BASE_DIR = os.getcwd()

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from math import sqrt


batch_size = 64
epochs=100
lr=0.001
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
neuron=128

#convert an array of values into a dataset matrix
def create_dataset(dataset, look_back):
    dataX, dataY = [], []

    for i in range(len(dataset) - look_back):
        dataX.append(dataset[i:i + look_back])
        dataY.append(dataset[i + look_back])

    return np.asarray(dataX), np.asarray(dataY)

# Separa a criação das entradas e dos alvos
def create_dataset_xy(X, y, look_back):
    dataX, dataY = [], []

    for i in range(len(X) - look_back):
        dataX.append(X[i:i + look_back])
        dataY.append(y[i + look_back])

    return np.asarray(dataX), np.asarray(dataY)

# Divide o conjunto de dados para a validação cruzada temporal e passa para o método.
# O método recebe o conjunto de dados da validação cruzada e a relação de divisão para etapa
# da validação.
# Ao final, salva os resultados médios das métricas da validação cruzada e do conjunto de dados.
def run_cv(method, datasets, dates, look_back, cap, n_splits=3, **kwargs):

    flag = kwargs.get('flag', '')

    tscv = TimeSeriesSplit(n_splits=n_splits)

    mape_list, classic_mape_list, rmse_list, mae_list = [], [], [], []
    y_test_all, y_pred_all, dates_all = [], [], []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(datasets)):

        logger.info(f"Executing Fold {fold + 1} - {flag}")

        # Dados de treino e teste do fold
        fold_idx = np.concatenate([train_idx, test_idx])

        datasets_fold = datasets[fold_idx]
        dates_fold = dates.iloc[fold_idx].reset_index(drop=True)
        split_ratio = len(train_idx) / len(datasets_fold)

        dates_pred = dates_fold.iloc[
            int(len(datasets_fold) * split_ratio) + look_back:
        ].reset_index(drop=True)

        y_test, y_pred = method(
            datasets=datasets_fold,
            look_back=look_back,
            data_partition=split_ratio,
            cap=cap,
            **kwargs
        )

        mask = y_test != 0
        mape = np.mean(np.abs(y_test - y_pred) / cap) * 100
        classic_mape = np.mean(
            np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])
        ) * 100
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        mae  = np.mean(np.abs(y_test - y_pred))

        mape_list.append(mape)
        classic_mape_list.append(classic_mape)
        rmse_list.append(rmse)
        mae_list.append(mae)

        y_test_all.extend(y_test)
        y_pred_all.extend(y_pred)
        dates_all.extend(dates_pred)

    assert len(y_test_all) == len(y_pred_all) == len(dates_all), (
        f"Tamanhos diferentes: "
        f"y_test={len(y_test_all)}, "
        f"y_pred={len(y_pred_all)}, "
        f"dates={len(dates_all)}"
    )

    # ============================================================
    # METRICAS DOS RESULTADOS
    # ============================================================

    metrics_summary = {
        "mape_mean": np.mean(mape_list),
        "mape_std": np.std(mape_list, ddof=1),

        "classic_mape_mean": np.mean(classic_mape_list),
        "classic_mape_std": np.std(classic_mape_list, ddof=1),

        "rmse_mean": np.mean(rmse_list),
        "rmse_std": np.std(rmse_list, ddof=1),

        "mae_mean": np.mean(mae_list),
        "mae_std": np.std(mae_list, ddof=1)
    }

    logger.info("\n===== CV RESULTS =====")
    logger.info("Model: %s", flag)

    for metric, value in metrics_summary.items():
        logger.info("%s: %.4f", metric, value)

    logger.info("")

    os.makedirs("predictions", exist_ok=True)

    np.savetxt(f'predictions/y_test_cv.txt',
           np.asarray(y_test_all).ravel(),
           fmt='%.6f')

    np.savetxt(f"predictions/y_pred_{flag}_cv.txt",
            np.asarray(y_pred_all).ravel(),
            fmt='%.6f')

    np.savetxt('predictions/dates_cv.txt',
            np.asarray(dates_all).astype(str),
            fmt='%s')

    return {
        "model": flag,

        "mape": metrics_summary["mape_mean"],
        "mape_std": metrics_summary["mape_std"],

        "classic_map": metrics_summary["classic_mape_mean"],
        "classic_map_std": metrics_summary["classic_mape_std"],

        "rmse": metrics_summary["rmse_mean"],
        "rmse_std": metrics_summary["rmse_std"],

        "mae": metrics_summary["mae_mean"],
        "mae_std": metrics_summary["mae_std"],

        "y_test": np.array(y_test_all),
        "y_pred": np.array(y_pred_all),
        "dates": np.array(dates_all)
    }


# Funcoes para o Transformer
# ===============================
# MultiHeadAttention manual (compatível TF 2.3)
# ===============================
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        assert embed_dim % num_heads == 0
        self.projection_dim = embed_dim // num_heads

        self.query_dense = tf.keras.layers.Dense(embed_dim)
        self.key_dense   = tf.keras.layers.Dense(embed_dim)
        self.value_dense = tf.keras.layers.Dense(embed_dim)

        self.combine_heads = tf.keras.layers.Dense(embed_dim)

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        return tf.matmul(weights, value)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        query = self.query_dense(inputs)
        key   = self.key_dense(inputs)
        value = self.value_dense(inputs)

        query = self.separate_heads(query, batch_size)
        key   = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention = self.attention(query, key, value)

        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))

        return self.combine_heads(concat_attention)

# ===============================
# Positional Encoding
# ===============================
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def call(self, x):

        length = tf.shape(x)[1]

        pos = tf.cast(tf.range(length), tf.float32)[:, tf.newaxis]
        i = tf.cast(tf.range(self.d_model), tf.float32)[tf.newaxis, :]

        angle_rates = 1.0 / tf.pow(
            10000.0,
            (2.0 * tf.floor(i / 2.0)) / float(self.d_model)
        )

        angles = pos * angle_rates

        angles_sin = tf.sin(angles)
        angles_cos = tf.cos(angles)

        pos_encoding = tf.where(
            tf.cast(tf.range(self.d_model) % 2, tf.bool),
            angles_cos,
            angles_sin
        )
        pos_encoding = tf.expand_dims(pos_encoding, axis=0)

        return x + pos_encoding


# =============================================================================
# DIRECT MODELS
# =============================================================================
# SVR
def svr_model(datasets, look_back, data_partition=0.7, cap=None, flag=''):

    datasets = np.asarray(datasets).reshape(-1, 1)

    train_size = int(len(datasets) * data_partition)

    train = datasets[:train_size]
    test = datasets[train_size:]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    sc_X = StandardScaler()
    sc_y = StandardScaler()

    X_train = sc_X.fit_transform(
        trainX.reshape(trainX.shape[0], -1)
    )
    X_test = sc_X.transform(
        testX.reshape(testX.shape[0], -1)
    )

    y_train = sc_y.fit_transform(
        np.asarray(trainY).reshape(-1, 1)
    ).ravel()

    model = SVR(kernel="rbf")
    model.fit(X_train, y_train)

    y_pred_scaled = model.predict(X_test)

    y_pred = sc_y.inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    ).ravel()

    y_test = np.asarray(testY).ravel()

    mask = y_test != 0

    mape = np.mean(np.abs(y_test - y_pred) / cap) * 100

    if np.any(mask):
        classic_mape = np.mean(
            np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])
        ) * 100
    else:
        classic_mape = np.nan

    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    mae = np.mean(np.abs(y_test - y_pred))

    logger.info(f"Classic MAPE: {classic_mape:.4f}%")
    logger.info(f"MAPE: {mape:.4f}%")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE: {mae:.4f}")

    return y_test, y_pred


# ANN
def ann_model(datasets, look_back, data_partition=0.7, cap=None, flag=''):

    # ==========================================================
    # PREPARAÇÃO DOS DADOS
    # ==========================================================
    datasets = np.asarray(datasets).reshape(-1, 1)

    train_size = int(len(datasets) * data_partition)

    train = datasets[:train_size]
    test = datasets[train_size:]

    # ==========================================================
    # CRIAÇÃO DAS SEQUÊNCIAS TEMPORAIS
    # ==========================================================
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    if len(trainX) == 0 or len(testX) == 0:
        raise ValueError(
            "Dados insuficientes para criar as sequências. "
            "Verifique look_back e o tamanho dos folds."
        )

    X_train = np.asarray(trainX)
    X_test = np.asarray(testX)

    y_train = np.asarray(trainY).reshape(-1, 1)
    y_test = np.asarray(testY).ravel()

    # ==========================================================
    # NORMALIZAÇÃO
    # ==========================================================
    sc_X = StandardScaler()
    sc_y = StandardScaler()

    X_train = sc_X.fit_transform(
        X_train.reshape(X_train.shape[0], -1)
    )

    X_test = sc_X.transform(
        X_test.reshape(X_test.shape[0], -1)
    )

    y_train = sc_y.fit_transform(y_train).reshape(-1, 1, 1)

    # ==========================================================
    # FORMATO DA ENTRADA DA ANN
    # ==========================================================
    X_train = X_train.reshape(
        X_train.shape[0], 1, X_train.shape[1]
    )

    X_test = X_test.reshape(
        X_test.shape[0], 1, X_test.shape[1]
    )

    # ==========================================================
    # MODELO ANN
    # ==========================================================
    model = Sequential()

    model.add(Dense(
        128,
        input_shape=(X_train.shape[1], X_train.shape[2])
    ))

    model.add(Dense(1))

    model.compile(
        loss="mse",
        optimizer=optimizer
    )

    model.fit(
        X_train,
        y_train,
        verbose=0
    )

    # ==========================================================
    # PREVISÃO
    # ==========================================================
    y_pred_scaled = model.predict(
        X_test,
        verbose=0
    ).ravel()

    y_pred = sc_y.inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    ).ravel()

    mask = y_test != 0

    mape = np.mean(np.abs(y_test - y_pred) / cap) * 100

    if np.any(mask):
        classic_mape = np.mean(
            np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])
        ) * 100
    else:
        classic_mape = np.nan

    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    mae = np.mean(np.abs(y_test - y_pred))

    logger.info(f"Classic MAPE: {classic_mape:.4f}%")
    logger.info(f"MAPE: {mape:.4f}%")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE: {mae:.4f}")

    return y_test, y_pred

##RF
def rf_model(new_data,i,look_back,data_partition,cap):
    
    x=i
    data1=new_data.loc[new_data['Month'].isin(x)]
    data1=data1.reset_index(drop=True)
    data1=data1.dropna()
    
    datas=data1['P_avg']
    datas_wind=pd.DataFrame(datas)
    dfs=datas
    s = dfs.values
    
    datasetss2=pd.DataFrame(s)
    datasets=datasetss2.values
    
    train_size = int(len(datasets) * data_partition)
    test_size = len(datasets) - train_size
    train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    X_train = pd.DataFrame(
        trainX.reshape(trainX.shape[0], -1)
    )
    Y_train = pd.DataFrame(trainY)

    X_test = pd.DataFrame(
        testX.reshape(testX.shape[0], -1)
    )
    Y_test = pd.DataFrame(testY)

    sc_X = StandardScaler()
    sc_y = StandardScaler()

    X = sc_X.fit_transform(X_train)
    y = sc_y.fit_transform(Y_train)

    X1 = sc_X.transform(X_test)
    y1 = sc_y.transform(Y_test)

    y = y.ravel()
    y1 = y1.ravel()
    
    numpy.random.seed(1234)
    tf.random.set_seed(1234)

    grid = RandomForestRegressor()
    grid.fit(X,y)
    y_pred_train_rf= grid.predict(X)
    y_pred_test_rf= grid.predict(X1)

    y_pred_train_rf=pd.DataFrame(y_pred_train_rf)
    y_pred_test_rf=pd.DataFrame(y_pred_test_rf)

    y1=pd.DataFrame(y1)
    y=pd.DataFrame(y)
 
        
    y_pred_test1_rf= sc_y.inverse_transform (y_pred_test_rf)
    y_pred_train1_rf=sc_y.inverse_transform (y_pred_train_rf)
   
    y_test= sc_y.inverse_transform (y1)
    y_train= sc_y.inverse_transform (y)
     
    y_pred_test1_rf=pd.DataFrame(y_pred_test1_rf)
    y_pred_train1_rf=pd.DataFrame(y_pred_train1_rf)
       
    y_test= pd.DataFrame(y_test)
        
    #summarize the fit of the model
    mask = y_test != 0
    mape=numpy.mean((numpy.abs(y_test-y_pred_test1_rf))/cap)*100
    classic_mape = np.mean(
        np.abs((y_test[mask] - y_pred_test1_rf[mask]) / y_test[mask])
    ) * 100
    rmse= sqrt(mean_squared_error(y_test,y_pred_test1_rf))
    mae=metrics.mean_absolute_error(y_test,y_pred_test1_rf)

    logger.info("MAPE: %s", mape)
    logger.info("Classic MAPE: %s", classic_mape)
    logger.info("RMSE: %s", rmse)
    logger.info("MAE: %s", mae)


##LSTM
def lstm_model(new_data,i,look_back,data_partition,cap):

    x=i
    data1=new_data.loc[new_data['Month'].isin(x)]
    data1=data1.reset_index(drop=True)
    data1=data1.dropna()
    
    datas=data1['P_avg']
    datas_wind=pd.DataFrame(datas)
    dfs=datas
    s = dfs.values
        
    datasetss2=pd.DataFrame(s)
    datasets=datasetss2.values
    
    train_size = int(len(datasets) * data_partition)
    test_size = len(datasets) - train_size
    train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    X_train = pd.DataFrame(
        trainX.reshape(trainX.shape[0], -1)
    )
    Y_train = pd.DataFrame(trainY)

    X_test = pd.DataFrame(
        testX.reshape(testX.shape[0], -1)
    )
    Y_test = pd.DataFrame(testY)

    sc_X = StandardScaler()
    sc_y = StandardScaler()

    X = sc_X.fit_transform(X_train)
    y = sc_y.fit_transform(Y_train)

    X1 = sc_X.transform(X_test)
    y1 = sc_y.transform(Y_test)

    y = y.ravel()
    y1 = y1.ravel()
    
    numpy.random.seed(1234)
    tf.random.set_seed(1234)


    trainX1 = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX1 = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))
    
    
    numpy.random.seed(1234)
    tf.random.set_seed(1234)

    model = Sequential()
    model.add(LSTM(units = neuron,input_shape=(trainX1.shape[1], trainX1.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse',optimizer=optimizer)

    model.fit(trainX1, y, epochs = epochs, batch_size=batch_size,verbose=0)
  # make predictions
    y_pred_train = model.predict(trainX1)
    y_pred_test = model.predict(testX1)
    y_pred_test= numpy.array(y_pred_test).ravel()

    y_pred_test=pd.DataFrame(y_pred_test)
    y_pred_test1= sc_y.inverse_transform (y_pred_test)
    y1=pd.DataFrame(y1)
      
    y_test= sc_y.inverse_transform (y1)
       
    #summarize the fit of the model
    mask = y_test != 0
    mape=numpy.mean((numpy.abs(y_test-y_pred_test1))/cap)*100
    classic_mape = np.mean(
        np.abs((y_test[mask] - y_pred_test1[mask]) / y_test[mask])
    ) * 100
    rmse= sqrt(mean_squared_error(y_test,y_pred_test1))
    mae=metrics.mean_absolute_error(y_test,y_pred_test1)

    
    logger.info("MAPE: %s", mape)
    logger.info("Classic MAPE: %s", classic_mape)
    logger.info("RMSE: %s", rmse)
    logger.info("MAE: %s", mae)


# In[24]:
# =============================================================================
# HYBRID MODELS
# =============================================================================

##HYBRID EMD LSTM
def emd_lstm(new_data,i,look_back,data_partition,cap):

    x=i
    data1=new_data.loc[new_data['Month'].isin(x)]
    data1=data1.reset_index(drop=True)
    data1=data1.dropna()
    
    datas=data1['P_avg']
    datas_wind=pd.DataFrame(datas)
    dfs=datas
    s = dfs.values
    
    emd = EMD()

    IMFs = emd(s)

    full_imf=pd.DataFrame(IMFs)
    data_decomp=full_imf.T

    pred_test=[]
    test_ori=[]
    pred_train=[]
    train_ori=[]

    for col in data_decomp:

        datasetss2=pd.DataFrame(data_decomp[col])
        datasets=datasetss2.values
        train_size = int(len(datasets) * data_partition)
        test_size = len(datasets) - train_size
        train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)

        X_train = pd.DataFrame(
            trainX.reshape(trainX.shape[0], -1)
        )
        Y_train = pd.DataFrame(trainY)

        X_test = pd.DataFrame(
            testX.reshape(testX.shape[0], -1)
        )
        Y_test = pd.DataFrame(testY)

        sc_X = StandardScaler()
        sc_y = StandardScaler()

        X = sc_X.fit_transform(X_train)
        y = sc_y.fit_transform(Y_train)

        X1 = sc_X.transform(X_test)
        y1 = sc_y.transform(Y_test)

        y = y.ravel()
        y1 = y1.ravel()

        trainX = numpy.reshape(X, (X.shape[0],X.shape[1],1))
        testX = numpy.reshape(X1, (X1.shape[0],X1.shape[1],1))
    
        numpy.random.seed(1234)
        tf.random.set_seed(1234)

        model = Sequential()
        model.add(LSTM(units = neuron,input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mse',optimizer=optimizer)

        # Fitting the RNN to the Training set
        model.fit(trainX, y, epochs = epochs, batch_size = batch_size,verbose=0)

        # make predictions
        y_pred_train = model.predict(trainX)
        y_pred_test = model.predict(testX)

        # make predictions

        y_pred_test= numpy.array(y_pred_test).ravel()
        y_pred_test=pd.DataFrame(y_pred_test)
        y1=pd.DataFrame(y1)
        y=pd.DataFrame(y)
        y_pred_train= numpy.array(y_pred_train).ravel()
        y_pred_train=pd.DataFrame(y_pred_train)

        y_test= sc_y.inverse_transform (y1)
        y_train= sc_y.inverse_transform (y)

        y_pred_test1= sc_y.inverse_transform (y_pred_test)
        y_pred_train1= sc_y.inverse_transform (y_pred_train)


        pred_test.append(y_pred_test1)
        test_ori.append(y_test)
        pred_train.append(y_pred_train1)
        train_ori.append(y_train)


    result_pred_test= pd.DataFrame.from_records(pred_test)
    result_pred_train= pd.DataFrame.from_records(pred_train)


    a=result_pred_test.sum(axis = 0, skipna = True) 
    b=result_pred_train.sum(axis = 0, skipna = True) 


    dataframe=pd.DataFrame(dfs)
    dataset=dataframe.values

    train_size = int(len(dataset) * data_partition)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    X_train = pd.DataFrame(
        trainX.reshape(trainX.shape[0], -1)
    )
    Y_train = pd.DataFrame(trainY)

    X_test = pd.DataFrame(
        testX.reshape(testX.shape[0], -1)
    )
    Y_test = pd.DataFrame(testY)

    sc_X = StandardScaler()
    sc_y = StandardScaler()

    X = sc_X.fit_transform(X_train)
    y = sc_y.fit_transform(Y_train)

    X1 = sc_X.transform(X_test)
    y1 = sc_y.transform(Y_test)

    y = y.ravel()
    y1 = y1.ravel()


    trainX = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    numpy.random.seed(1234)
    tf.random.set_seed(1234)
    
    y1=pd.DataFrame(y1)
    y=pd.DataFrame(y)

    y_test= sc_y.inverse_transform (y1)
    y_train= sc_y.inverse_transform (y)

    a= pd.DataFrame(a)    
    y_test= pd.DataFrame(y_test)    

    #summarize the fit of the model
    mask = y_test != 0
    mape=numpy.mean((numpy.abs(y_test-a))/cap)*100
    classic_mape = np.mean(
        np.abs((y_test[mask] - a[mask]) / y_test[mask])
    ) * 100
    rmse= sqrt(mean_squared_error(y_test,a))
    mae=metrics.mean_absolute_error(y_test,a)

    
    logger.info("MAPE: %s", mape)
    logger.info("Classic MAPE: %s", classic_mape)
    logger.info("RMSE: %s", rmse)
    logger.info("MAE: %s", mae)


# In[25]:


##HYBRID EEMD LSTM
def eemd_lstm(new_data,i,look_back,data_partition,cap):

    x=i
    data1=new_data.loc[new_data['Month'].isin(x)]
    data1=data1.reset_index(drop=True)
    data1=data1.dropna()
    
    datas=data1['P_avg']
    datas_wind=pd.DataFrame(datas)
    dfs=datas
    s = dfs.values
    
    emd = EEMD(noise_width=0.02)
    emd.noise_seed(12345)

    IMFs = emd(s)

    full_imf=pd.DataFrame(IMFs)
    data_decomp=full_imf.T
    


    pred_test=[]
    test_ori=[]
    pred_train=[]
    train_ori=[]

    for col in data_decomp:

        datasetss2=pd.DataFrame(data_decomp[col])
        datasets=datasetss2.values
        train_size = int(len(datasets) * data_partition)
        test_size = len(datasets) - train_size
        train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)

        X_train = pd.DataFrame(
            trainX.reshape(trainX.shape[0], -1)
        )
        Y_train = pd.DataFrame(trainY)

        X_test = pd.DataFrame(
            testX.reshape(testX.shape[0], -1)
        )
        Y_test = pd.DataFrame(testY)

        sc_X = StandardScaler()
        sc_y = StandardScaler()

        X = sc_X.fit_transform(X_train)
        y = sc_y.fit_transform(Y_train)

        X1 = sc_X.transform(X_test)
        y1 = sc_y.transform(Y_test)

        y = y.ravel()
        y1 = y1.ravel()

        trainX = numpy.reshape(X, (X.shape[0],X.shape[1],1))
        testX = numpy.reshape(X1, (X1.shape[0],X1.shape[1],1))

        numpy.random.seed(1234)
        tf.random.set_seed(1234)

        model = Sequential()
        model.add(LSTM(units = neuron,input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mse',optimizer=optimizer)

        # Fitting the RNN to the Training set
        model.fit(trainX, y, epochs = epochs, batch_size = batch_size,verbose=0)

        # make predictions
        y_pred_train = model.predict(trainX)
        y_pred_test = model.predict(testX)

        # make predictions
        y_pred_test= numpy.array(y_pred_test).ravel()
        y_pred_test=pd.DataFrame(y_pred_test)
        y1=pd.DataFrame(y1)
        y=pd.DataFrame(y)
        y_pred_train= numpy.array(y_pred_train).ravel()
        y_pred_train=pd.DataFrame(y_pred_train)

        y_test= sc_y.inverse_transform (y1)
        y_train= sc_y.inverse_transform (y)

        y_pred_test1= sc_y.inverse_transform (y_pred_test)
        y_pred_train1= sc_y.inverse_transform (y_pred_train)

        pred_test.append(y_pred_test1)
        test_ori.append(y_test)
        pred_train.append(y_pred_train1)
        train_ori.append(y_train)


    result_pred_test= pd.DataFrame.from_records(pred_test)
    result_pred_train= pd.DataFrame.from_records(pred_train)


    a=result_pred_test.sum(axis = 0, skipna = True) 
    b=result_pred_train.sum(axis = 0, skipna = True) 

    dataframe=pd.DataFrame(dfs)
    dataset=dataframe.values

    train_size = int(len(dataset) * data_partition)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    X_train = pd.DataFrame(
        trainX.reshape(trainX.shape[0], -1)
    )
    Y_train = pd.DataFrame(trainY)

    X_test = pd.DataFrame(
        testX.reshape(testX.shape[0], -1)
    )
    Y_test = pd.DataFrame(testY)

    sc_X = StandardScaler()
    sc_y = StandardScaler()

    X = sc_X.fit_transform(X_train)
    y = sc_y.fit_transform(Y_train)

    X1 = sc_X.transform(X_test)
    y1 = sc_y.transform(Y_test)

    y = y.ravel()
    y1 = y1.ravel()

    trainX = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    numpy.random.seed(1234)
    tf.random.set_seed(1234)
    
    y1=pd.DataFrame(y1)
    y=pd.DataFrame(y)

    y_test= sc_y.inverse_transform (y1)
    y_train= sc_y.inverse_transform (y)

    a= pd.DataFrame(a)    
    y_test= pd.DataFrame(y_test)    

    #summarize the fit of the model
    mask = y_test != 0
    mape=numpy.mean((numpy.abs(y_test-a))/cap)*100
    classic_mape = np.mean(
        np.abs((y_test[mask] - a[mask]) / y_test[mask])
    ) * 100
    rmse= sqrt(mean_squared_error(y_test,a))
    mae=metrics.mean_absolute_error(y_test,a)

    
    logger.info("MAPE: %s", mape)
    logger.info("Classic MAPE: %s", classic_mape)
    logger.info("RMSE: %s", rmse)
    logger.info("MAE: %s", mae)


##HYBRID CEEMDAN LSTM
def ceemdan_lstm(new_data,i,look_back,data_partition,cap):

    x=i
    data1=new_data.loc[new_data['Month'].isin(x)]
    data1=data1.reset_index(drop=True)
    data1=data1.dropna()
    
    datas=data1['P_avg']
    datas_wind=pd.DataFrame(datas)
    dfs=datas
    s = dfs.values
    
    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)

    IMFs = emd(s)

    full_imf=pd.DataFrame(IMFs)
    data_decomp=full_imf.T
    


    pred_test=[]
    test_ori=[]
    pred_train=[]
    train_ori=[]

    for col in data_decomp:

        datasetss2=pd.DataFrame(data_decomp[col])
        datasets=datasetss2.values
        train_size = int(len(datasets) * data_partition)
        test_size = len(datasets) - train_size
        train, test = datasets[0:train_size], datasets[train_size:len(datasets)]
    
        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)

        X_train = pd.DataFrame(
            trainX.reshape(trainX.shape[0], -1)
        )
        Y_train = pd.DataFrame(trainY)

        X_test = pd.DataFrame(
            testX.reshape(testX.shape[0], -1)
        )
        Y_test = pd.DataFrame(testY)

        sc_X = StandardScaler()
        sc_y = StandardScaler()

        X = sc_X.fit_transform(X_train)
        y = sc_y.fit_transform(Y_train)

        X1 = sc_X.transform(X_test)
        y1 = sc_y.transform(Y_test)

        y = y.ravel()
        y1 = y1.ravel()

        trainX = numpy.reshape(X, (X.shape[0],X.shape[1],1))
        testX = numpy.reshape(X1, (X1.shape[0],X1.shape[1],1))

        numpy.random.seed(1234)
        tf.random.set_seed(1234)

        model = Sequential()
        model.add(LSTM(units = neuron,input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mse',optimizer=optimizer)


        # Fitting the RNN to the Training set
        model.fit(trainX, y, epochs = epochs, batch_size = batch_size,verbose=0)

        # make predictions
        y_pred_train = model.predict(trainX)
        y_pred_test = model.predict(testX)

        # make predictions

        y_pred_test= numpy.array(y_pred_test).ravel()
        y_pred_test=pd.DataFrame(y_pred_test)
        y1=pd.DataFrame(y1)
        y=pd.DataFrame(y)
        y_pred_train= numpy.array(y_pred_train).ravel()
        y_pred_train=pd.DataFrame(y_pred_train)

        y_test= sc_y.inverse_transform (y1)
        y_train= sc_y.inverse_transform (y)

        y_pred_test1= sc_y.inverse_transform (y_pred_test)
        y_pred_train1= sc_y.inverse_transform (y_pred_train)


        pred_test.append(y_pred_test1)
        test_ori.append(y_test)
        pred_train.append(y_pred_train1)
        train_ori.append(y_train)


    result_pred_test= pd.DataFrame.from_records(pred_test)
    result_pred_train= pd.DataFrame.from_records(pred_train)


    a=result_pred_test.sum(axis = 0, skipna = True) 
    b=result_pred_train.sum(axis = 0, skipna = True) 


    dataframe=pd.DataFrame(dfs)
    dataset=dataframe.values

    train_size = int(len(dataset) * data_partition)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    X_train = pd.DataFrame(
        trainX.reshape(trainX.shape[0], -1)
    )
    Y_train = pd.DataFrame(trainY)

    X_test = pd.DataFrame(
        testX.reshape(testX.shape[0], -1)
    )
    Y_test = pd.DataFrame(testY)

    sc_X = StandardScaler()
    sc_y = StandardScaler()

    X = sc_X.fit_transform(X_train)
    y = sc_y.fit_transform(Y_train)

    X1 = sc_X.transform(X_test)
    y1 = sc_y.transform(Y_test)

    y = y.ravel()
    y1 = y1.ravel()

    trainX = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    numpy.random.seed(1234)
    tf.random.set_seed(1234)
    
    y1=pd.DataFrame(y1)
    y=pd.DataFrame(y)

    y_test= sc_y.inverse_transform (y1)
    y_train= sc_y.inverse_transform (y)

    a= pd.DataFrame(a)    
    y_test= pd.DataFrame(y_test)    

    #summarize the fit of the model
    mask = y_test != 0
    mape=numpy.mean((numpy.abs(y_test-a))/cap)*100
    classic_mape = np.mean(
        np.abs((y_test[mask] - a[mask]) / y_test[mask])
    ) * 100
    rmse= sqrt(mean_squared_error(y_test,a))
    mae=metrics.mean_absolute_error(y_test,a)

    
    logger.info("MAPE: %s", mape)
    logger.info("Classic MAPE: %s", classic_mape)
    logger.info("RMSE: %s", rmse)
    logger.info("MAE: %s", mae)


##Proposed Method Hybrid CEEMDAN-EWT LSTM
def proposed_method(
        datasets, look_back, data_partition, cap,
        lstm=True, gru=False, drop_extend=False, bilstm=False,
        bigru=False, transformer=False, hilbert=False, flag='', save=True,
    ):

    # ============================================================
    # 1. DECOMPOSIÇÃO CEEMDAN
    # ============================================================

    s = np.ascontiguousarray(
        np.asarray(datasets, dtype=np.float64).reshape(-1)
    )

    emd_ = CEEMDAN(
        # trials=10,
        epsilon=0.05,
        parallel=False
    )

    emd_.noise_seed(12345)

    IMFs = emd_.ceemdan(s)

    logger.info("Shape IMFs: %s", IMFs.shape)

    full_imf = pd.DataFrame(IMFs)
    ceemdan1 = full_imf.T

    # ============================================================
    # 2. APLICAÇÃO DA EWT NA PRIMEIRA IMF
    # ============================================================

    imf1 = ceemdan1.iloc[:, 0]

    ewt, mfb, boundaries = ewtpy.EWT1D(imf1, N=3)

    df_ewt = pd.DataFrame(ewt)

    df_ewt.drop(df_ewt.columns[2], axis=1, inplace=True)

    denoised = df_ewt.sum(axis=1, skipna=True)

    ceemdan_without_imf1 = ceemdan1.iloc[:, 1:]

    new_ceemdan = pd.concat(
        [denoised, ceemdan_without_imf1],
        axis=1
    )

    logger.info("Componentes para previsão: %s", new_ceemdan.shape[1])

    for i in range(new_ceemdan.shape[1]):

        component = new_ceemdan.iloc[:, i].to_numpy()

        logger.info(
            f"Componente {i + 1}: "
            f"mínimo={np.min(component):.8f}, " 
            f"máximo={np.max(component):.8f}, "
            f"desvio={np.std(component):.8f}, "
            f"NaN={np.isnan(component).sum()}, "
            f"Inf={np.isinf(component).sum()}"
        )

    # ============================================================
    # 3. PREPARAÇÃO DAS ENTRADAS
    # ============================================================
    # Após a IMF 9 os valores são muito constantes. Dessa forma, não é possível calcular a
    # HHT. Optou-se, neste cenário, por utilizar a própria componente como entrada sem a transformação.
    # Também poderiam ser utilizadas apenas a HHT das componente de 1 a 9.
    if hilbert:

        logger.info("Aplicando HHT...")

        sample_rate = 1.0    # Verificar

        IP, IF, IA = emd.spectra.frequency_transform(
            new_ceemdan.values,
            sample_rate,
            'nht'
        )

        freq_edges, freq_centres = emd.spectra.define_hist_bins(
            0,
            sample_rate / 2,
            64,
            'linear'
        )

        hht_features = []
        
        for i in range(new_ceemdan.shape[1]):

            component_if = IF[:, i:i+1].copy()
            component_ia = IA[:, i:i+1].copy()

            # Verifica se existem valores válidos na amplitude instantânea
            valid_ia = np.isfinite(component_ia)

            if not np.any(valid_ia):

                logger.info(
                    f"Componente {i + 1}: "
                    "HHT não aplicável. Utilizando componente original.",
                )

                hht_component = new_ceemdan.iloc[:, i].to_numpy(
                    dtype=np.float32
                ).reshape(-1, 1)

            else:

                invalid = (
                    ~np.isfinite(component_if) |
                    ~np.isfinite(component_ia)
                )

                component_if[invalid] = 0.0
                component_ia[invalid] = 0.0

                f, spectrum = emd.spectra.hilberthuang(
                    component_if,
                    component_ia,
                    freq_edges,
                    sum_time=False,
                    sum_imfs=True,
                    mode='amplitude'
                )

                hht_component = np.asarray(
                    spectrum.T,
                    dtype=np.float32
                )

                hht_component = np.nan_to_num(
                    hht_component,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0
                )

            hht_features.append(hht_component)

        input_components = hht_features

    else:

        logger.info("Utilizando componentes originais como entradas...")

        input_components = [
            new_ceemdan.iloc[:, i].to_numpy(
                dtype=np.float32
            ).reshape(-1, 1)
            for i in range(new_ceemdan.shape[1])
        ]

    # ============================================================
    # 4. PREVISÃO INDIVIDUAL DE CADA COMPONENTE
    # ============================================================
    logger.info("Inicio do treinamento do modelo")

    pred_test = []
    pred_train = []

    for idx, col in enumerate(new_ceemdan.columns):

        logger.info(f"\nProcessando componente {idx + 1}")

        # Entrada: HHT ou componente original
        features = np.asarray(
            input_components[idx],
            dtype=np.float32
        )

        # Alvo: sempre a componente original
        target = new_ceemdan[col].to_numpy(
            dtype=np.float32
        )

        # Verifica alinhamento temporal
        if features.shape[0] != len(target):
            raise ValueError(
                f"Erro de dimensão na componente {idx + 1}: "
                f"features={features.shape}, target={target.shape}"
            )

        # --------------------------------------------------------
        # Separação treino/teste
        # --------------------------------------------------------

        train_size = int(len(target) * data_partition)

        X_train_raw = features[:train_size]
        X_test_raw = features[train_size:]

        y_train_raw = target[:train_size]
        y_test_raw = target[train_size:]

        # --------------------------------------------------------
        # Criação das janelas temporais
        # --------------------------------------------------------

        trainX, trainY = create_dataset_xy(
            X_train_raw,
            y_train_raw,
            look_back
        )

        testX, testY = create_dataset_xy(
            X_test_raw,
            y_test_raw,
            look_back
        )

        trainX = np.asarray(trainX, dtype=np.float32)
        testX = np.asarray(testX, dtype=np.float32)

        trainY = np.asarray(trainY, dtype=np.float32).reshape(-1, 1)
        testY = np.asarray(testY, dtype=np.float32).reshape(-1, 1)

        # --------------------------------------------------------
        # Normalização
        # --------------------------------------------------------

        sc_X = StandardScaler()
        sc_y = StandardScaler()

        n_features = trainX.shape[2]

        X = sc_X.fit_transform(
            trainX.reshape(-1, n_features)
        ).reshape(trainX.shape)

        X1 = sc_X.transform(
            testX.reshape(-1, n_features)
        ).reshape(testX.shape)

        y = sc_y.fit_transform(trainY)
        y1 = sc_y.transform(testY)

        # Formato: (amostras, look_back, número de features)
        trainX = np.ascontiguousarray(X, dtype=np.float32)
        testX = np.ascontiguousarray(X1, dtype=np.float32)

        y = np.ascontiguousarray(y.reshape(-1, 1), dtype=np.float32)
        y1 = np.ascontiguousarray(y1.reshape(-1, 1), dtype=np.float32)

        logger.info(
            f"trainX: {trainX.shape} | "
            f"testX: {testX.shape} | "
            f"Features: {n_features}"
        )

        # --------------------------------------------------------
        # Construção do modelo
        # --------------------------------------------------------

        K.clear_session()
        gc.collect()

        np.random.seed(1234)
        tf.random.set_seed(1234)

        if transformer:
            inputs = tf.keras.Input(
                shape=(trainX.shape[1], trainX.shape[2])
            )

            x = tf.keras.layers.Dense(32)(inputs)
            x = PositionalEncoding(32)(x)

            attn = MultiHeadAttention(
                embed_dim=32,
                num_heads=4
            )(x)

            attn = Dropout(0.1)(attn)
            x = tf.keras.layers.Add()([x, attn])
            x = tf.keras.layers.LayerNormalization()(x)

            ff = Dense(64, activation="relu")(x)
            ff = Dense(32)(ff)
            ff = Dropout(0.2)(ff)

            x = tf.keras.layers.Add()([x, ff])
            x = tf.keras.layers.LayerNormalization()(x)

            x = tf.keras.layers.Lambda(lambda t: t[:, -1, :])(x)

            outputs = Dense(1)(x)

            model = tf.keras.Model(inputs=inputs, outputs=outputs)

        else:
            model = Sequential()

            if gru:
                model.add(GRU(
                    units=neuron,
                    input_shape=(trainX.shape[1], trainX.shape[2])
                    )
                )

            elif lstm:
                model.add(LSTM(
                        units=neuron,
                        input_shape=(trainX.shape[1], trainX.shape[2])
                    )
                )

            elif bilstm:
                model.add(
                    Bidirectional(
                        LSTM(units=neuron),
                        input_shape=(trainX.shape[1], trainX.shape[2])
                    )
                )

            elif bigru:
                model.add(
                    Bidirectional(
                        GRU(units=neuron),
                        input_shape=(trainX.shape[1], trainX.shape[2])
                    )
                )

            # Camadas adicionais
            if drop_extend:
                model.add(Dense(64, activation='relu'))
                model.add(Dropout(0.2))
            
            # Camada de saída
            model.add(Dense(1))


        model.compile(
            loss="mse",
            optimizer=optimizer,
            run_eagerly=True
        )

        # --------------------------------------------------------
        # Treinamento
        # --------------------------------------------------------

        model.fit(
            trainX,
            y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )

        # --------------------------------------------------------
        # Previsões
        # --------------------------------------------------------

        y_pred_train = model.predict(trainX, verbose=0)
        y_pred_test = model.predict(testX, verbose=0)

        # Retorna à escala original
        y_pred_train1 = sc_y.inverse_transform(
            np.asarray(y_pred_train).reshape(-1, 1)
        )

        y_pred_test1 = sc_y.inverse_transform(
            np.asarray(y_pred_test).reshape(-1, 1)
        )

        pred_train.append(y_pred_train1)
        pred_test.append(y_pred_test1)

        # Libera recursos
        del model

        K.clear_session()
        gc.collect()

    # ============================================================
    # 5. RECONSTRUÇÃO DO SINAL
    # ============================================================

    # Cada componente deve possuir uma previsão por instante.
    pred_test_array = np.stack(
        [np.asarray(p).reshape(-1) for p in pred_test],
        axis=0
    )

    pred_train_array = np.stack(
        [np.asarray(p).reshape(-1) for p in pred_train],
        axis=0
    )

    # Reconstrução pela soma das componentes
    a = np.sum(pred_test_array, axis=0)
    b = np.sum(pred_train_array, axis=0)

    logger.info("Shape reconstrução teste: %s", a.shape)
    logger.info("Shape reconstrução treino: %s", b.shape)

    # ============================================================
    # 6. VALORES REAIS DO SINAL ORIGINAL
    # ============================================================

    datasets = np.asarray(datasets).reshape(-1)

    train_size = int(len(datasets) * data_partition)

    train = datasets[:train_size]
    test = datasets[train_size:]

    # Alinhamento com as previsões
    y_test = test[look_back:look_back + len(a)]
    y_train = train[look_back:look_back + len(b)]

    # ============================================================
    # 7. MÉTRICAS
    # ============================================================

    mask = y_test != 0

    mape = np.mean(
        np.abs(y_test - a)
    ) / cap * 100

    classic_mape = np.mean(
        np.abs((y_test[mask] - a[mask]) / y_test[mask])
    ) * 100

    rmse = sqrt(
        mean_squared_error(y_test, a)
    )

    mae = metrics.mean_absolute_error(
        y_test, a
    )

    logger.info("MAPE: %s", mape)
    logger.info("Classic MAPE: %s", classic_mape)
    logger.info("RMSE: %s", rmse)
    logger.info("MAE: %s", mae)

    return y_test.flatten(), a.flatten()


# Proposed Method Hybrid CEEMDAN (without EWT) with Hilbert transform. This is a variation
# of the Hilbert-Huang Transformer (HHT)

# def proposed_method_hilbert_transform(new_data, i, look_back, data_partition, cap):

#     # =========================
#     # 1. Data preparation
#     # =========================
#     data1 = new_data.loc[new_data['Month'].isin(i)]
#     data1 = data1.reset_index(drop=True).dropna()

#     dfs = data1['P_avg']
#     s = dfs.values

#     # =========================
#     # 2. CEEMDAN decomposition
#     # =========================
#     ceemdan = CEEMDAN(epsilon=0.05)
#     ceemdan.noise_seed(12345)

#     IMFs = ceemdan(s)
#     ceemdan_df = pd.DataFrame(IMFs).T  # (N, num_imfs)

#     # =========================
#     # 3. Hilbert Transform
#     # =========================
#     imfs_array = ceemdan_df.values

#     analytic = hilbert(imfs_array, axis=0)
#     amplitude = np.abs(analytic)

#     # features (estável)
#     features = np.concatenate([imfs_array, amplitude], axis=1)
#     new_features = pd.DataFrame(features)

#     # =========================
#     # 4. LSTM por feature
#     # =========================
#     pred_test = []

#     for col in new_features:

#         # sempre 2D
#         datasets = new_features[[col]].values

#         train_size = int(len(datasets) * data_partition)
#         train, test = datasets[:train_size], datasets[train_size:]

#         trainX, trainY = create_dataset(train, look_back)
#         testX, testY = create_dataset(test, look_back)

#         # garantir formato correto
#         trainY = np.array(trainY).reshape(-1, 1)
#         testY  = np.array(testY).reshape(-1, 1)

#         sc_X = StandardScaler()
#         sc_y = StandardScaler()

#         # FIT apenas no treino
#         X_train = sc_X.fit_transform(trainX)
#         y_train = sc_y.fit_transform(trainY)

#         # TEST usa transform
#         X_test = sc_X.transform(testX)
#         y_test_scaled = sc_y.transform(testY)

#         # reshape para LSTM
#         trainX = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
#         testX  = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

#         # reprodutibilidade
#         np.random.seed(1234)
#         tf.random.set_seed(1234)

#         # modelo
#         model = Sequential()
#         model.add(LSTM(128, input_shape=(trainX.shape[1], 1)))
#         model.add(Dense(1))

#         model.fit(trainX, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

#         # previsões
#         y_pred_test = model.predict(testX)

#         # voltar escala original
#         y_pred_test = sc_y.inverse_transform(y_pred_test)

#         pred_test.append(y_pred_test)

#         K.clear_session()
#         del model
#         gc.collect()

#     # =========================
#     # 5. Reconstrução
#     # =========================
#     result_pred_test = pd.DataFrame.from_records(pred_test)
#     a = result_pred_test.sum(axis=0, skipna=True)

#     # =========================
#     # 6. Ground truth
#     # =========================
#     dataset = dfs.values.reshape(-1, 1)

#     train_size = int(len(dataset) * data_partition)
#     train, test = dataset[:train_size], dataset[train_size:]

#     _, testY = create_dataset(test, look_back)

#     # corrigir shape
#     testY = np.array(testY).reshape(-1, 1)

#     y_test = pd.DataFrame(testY)
#     a = pd.DataFrame(a)

#     # =========================
#     # 7. Métricas
#     # =========================
#     mape = np.mean(np.abs(y_test - a) / cap) * 100
#     rmse = sqrt(mean_squared_error(y_test, a))
#     mae = metrics.mean_absolute_error(y_test, a)

#     logger.info('MAPE', mape)
#     logger.info('RMSE', rmse)
#     logger.info('MAE', mae)

#     # 8. Save the results
#     np.savetxt(os.path.join(BASE_DIR, 'y_test.txt'),
#                 y_test.values.flatten(), fmt='%.6f')
#     np.savetxt(os.path.join(BASE_DIR, 'y_pred_proposed_method_ht.txt'),
#                 a.values.flatten(), fmt='%.6f')

#     dates = data1['Month'].iloc[-len(y_test):]
#     dates = pd.to_datetime(dates)

#     np.savetxt(os.path.join(BASE_DIR, 'dates.txt'),
#                 dates.astype(str), fmt='%s')


#     return a, y_test


# def proposed_method_hilbert_transform(datasets, look_back, data_partition, cap):

#     # Prepara o sinal para o CEEMDAN
#     s = np.ascontiguousarray(
#         np.asarray(datasets, dtype=np.float64).reshape(-1)
#     )

#     emd = CEEMDAN(
#         trials=10,
#         epsilon=0.05,
#         parallel=False
#     )

#     emd.noise_seed(12345)

#     IMFs = emd.ceemdan(s)
#     ceemdan_df = pd.DataFrame(IMFs).T

#     imfs_array = ceemdan_df.values

#     # =========================
#     # 3. Hilbert Transform
#     # =========================
#     analytic = hilbert(imfs_array, axis=0)

#     amplitude = np.abs(analytic)
#     phase = np.unwrap(np.angle(analytic), axis=0)

#     inst_freq = np.diff(phase, axis=0)
#     inst_freq = np.vstack([inst_freq, np.zeros((1, inst_freq.shape[1]))])

#     # =========================
#     # 4. Features finais
#     # =========================
#     features = np.concatenate([
#         imfs_array,
#         amplitude,
#         inst_freq
#     ], axis=1)

#     # =========================
#     # 5. Dataset supervisionado
#     # =========================
#     def create_multivariate_dataset(X, y, look_back):
#         Xs, ys = [], []
#         for i in range(len(X) - look_back):
#             Xs.append(X[i:i+look_back, :])
#             ys.append(y[i+look_back])
#         return np.array(Xs), np.array(ys)

#     X_all, y_all = create_multivariate_dataset(features, y_series, look_back)

#     # =========================
#     # 6. Split
#     # =========================
#     train_size = int(len(X_all) * data_partition)

#     X_train, X_test = X_all[:train_size], X_all[train_size:]
#     y_train, y_test = y_all[:train_size], y_all[train_size:]

#     # =========================
#     # 7. Scaling
#     # =========================
#     sc_X = StandardScaler()
#     sc_y = StandardScaler()

#     # reshape para scaler
#     X_train_2d = X_train.reshape(-1, X_train.shape[2])
#     X_test_2d  = X_test.reshape(-1, X_test.shape[2])

#     X_train_scaled = sc_X.fit_transform(X_train_2d)
#     X_test_scaled  = sc_X.transform(X_test_2d)

#     # volta para 3D
#     X_train = X_train_scaled.reshape(X_train.shape)
#     X_test  = X_test_scaled.reshape(X_test.shape)

#     y_train = sc_y.fit_transform(y_train)
#     y_test_scaled = sc_y.transform(y_test)

#     # =========================
#     # 8. Modelo LSTM
#     # =========================
#     np.random.seed(1234)
#     tf.random.set_seed(1234)

#     model = Sequential()
#     model.add(LSTM(128, input_shape=(look_back, X_train.shape[2])))
#     model.add(Dense(activation='relu'))
#     model.add(Dense(1))


#     model.fit(X_train, y_train,
#               epochs=epochs,
#               batch_size=batch_size,
#               verbose=1)

#     # =========================
#     # 9. Previsão
#     # =========================
#     y_pred = model.predict(X_test)

#     # voltar escala
#     y_pred = sc_y.inverse_transform(y_pred)
#     y_test = sc_y.inverse_transform(y_test_scaled)

#     dates = data1['Month'].iloc[-len(y_test):]
#     dates = pd.to_datetime(dates)

#     # =========================
#     # 10. Métricas
#     # =========================
#     mask = y_test != 0
#     mape = np.mean(np.abs(y_test - y_pred) / cap) * 100
#     classic_mape = np.mean(
#         np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])
#     ) * 100
#     rmse = sqrt(mean_squared_error(y_test, y_pred))
#     mae = metrics.mean_absolute_error(y_test, y_pred)

#     logger.info('MAPE', mape)
#     logger.info('Classic MAPE', classic_mape)
#     logger.info('RMSE', rmse)
#     logger.info('MAE', mae)

#     # Save metrics
#     try:
#         BASE_DIR = "/app"
#         np.savetxt(os.path.join(BASE_DIR, 'y_test_hht.txt'),
#                    y_test.flatten(), fmt='%.6f')
#         np.savetxt(os.path.join(BASE_DIR, 'y_pred_hht.txt'),
#                    y_pred.flatten(), fmt='%.6f')
#     except:
#         pass

#     return y_test.flatten(), y_pred.flatten(), dates.astype(str)


if __name__=='__main__':
    df = pd.read_csv('dataset/final_la_haute_R0711.csv')

    df['Date'] = pd.to_datetime(df['Date_time'], dayfirst=True)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    new_data = df[['Month', 'Year', 'Date', 'P_avg']]
    new_data = new_data[new_data.Year == 2017]

    cap = max(new_data['P_avg'])

    i = [1]
    look_back = 6
    data_partition = 0.8

    data1 = new_data[new_data['Month'].isin(i)].reset_index(drop=True)
    datasets = data1['P_avg'].to_numpy(dtype=np.float64)

    proposed_method(
        datasets=datasets,
        look_back=look_back,
        data_partition=data_partition,
        cap=cap
    )