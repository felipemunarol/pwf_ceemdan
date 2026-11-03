#!/usr/bin/env python
# coding: utf-8

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["AUTOGRAPH_VERBOSITY"] = "0"
os.environ["PYTHONHASHSEED"] = "1234"

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
epochs=10
lr=0.001
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
neuron=128

#convert an array of values into a dataset matrix
def create_dataset(X, y, look_back):
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

    tscv = TimeSeriesSplit(n_splits=n_splits)

    mape_list, classic_mape_list, rmse_list, mae_list = [], [], [], []
    y_test_all, y_pred_all, dates_all = [], [], []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(datasets)):

        print(f"Executing Fold {fold + 1} - {method.__name__}")

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

    print("\n===== CV RESULTS =====")
    print("Model:", method.__name__)
    print("Classic MAPE:", np.mean(classic_mape_list))
    print("MAPE:", np.mean(mape_list))
    print("RMSE:", np.mean(rmse_list))
    print("MAE :", np.mean(mae_list))

    np.savetxt(f'y_test_cv.txt',
           np.asarray(y_test_all).ravel(),
           fmt='%.6f')

    np.savetxt(f'y_pred_{method.__name__}_cv.txt',
            np.asarray(y_pred_all).ravel(),
            fmt='%.6f')

    np.savetxt('dates_cv.txt',
            np.asarray(dates_all).astype(str),
            fmt='%s')

    return {
        "model": method.__name__,
        "mape": np.mean(mape_list),
        "classic_map": np.mean(classic_mape_list),
        "rmse": np.mean(rmse_list),
        "mae": np.mean(mae_list),
        "y_test": np.array(y_test_all),
        "y_pred": np.array(y_pred_all),
        "dates": np.array(dates_all)
    }

# =============================================================================
# DIRECT MODELS
# =============================================================================
# SVR
def svr_model(datasets, look_back, data_partition=0.7, cap=None):

    datasets = np.asarray(datasets).reshape(-1, 1)

    train_size = int(len(datasets) * data_partition)

    train = datasets[:train_size]
    test = datasets[train_size:]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    sc_X = StandardScaler()
    sc_y = StandardScaler()

    X_train = sc_X.fit_transform(trainX)
    X_test = sc_X.transform(testX)

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

    return y_test, y_pred


# ANN
def ann_model(datasets, look_back, data_partition=0.7, cap=None):

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

    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    y_train = sc_y.fit_transform(y_train).ravel()

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
    X_train=pd.DataFrame(trainX)
    Y_train=pd.DataFrame(trainY)
    X_test=pd.DataFrame(testX)
    Y_test=pd.DataFrame(testY)
    sc_X = StandardScaler()
    sc_y = StandardScaler()

    X= sc_X.fit_transform(X_train)
    y= sc_y.fit_transform(Y_train)
    X1= sc_X.fit_transform(X_test)
    y1= sc_y.fit_transform(Y_test)
    y=y.ravel()
    y1=y1.ravel()  
    
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

    print('MAPE',mape)
    print('Classic MAPE', classic_mape)
    print('RMSE',rmse)
    print('MAE',mae)


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
    X_train=pd.DataFrame(trainX)
    Y_train=pd.DataFrame(trainY)
    X_test=pd.DataFrame(testX)
    Y_test=pd.DataFrame(testY)
    sc_X = StandardScaler()
    sc_y = StandardScaler()

    X= sc_X.fit_transform(X_train)
    y= sc_y.fit_transform(Y_train)
    X1= sc_X.fit_transform(X_test)
    y1= sc_y.fit_transform(Y_test)
    y=y.ravel()
    y1=y1.ravel()  
    
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

    
    print('MAPE',mape)
    print('Classic MAPE', classic_mape)
    print('RMSE',rmse)
    print('MAE',mae)


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
        X_train=pd.DataFrame(trainX)
        Y_train=pd.DataFrame(trainY)
        X_test=pd.DataFrame(testX)
        Y_test=pd.DataFrame(testY)
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X= sc_X.fit_transform(X_train)
        y= sc_y.fit_transform(Y_train)
        X1= sc_X.fit_transform(X_test)
        y1= sc_y.fit_transform(Y_test)
        y=y.ravel()
        y1=y1.ravel()  

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
    X_train=pd.DataFrame(trainX)
    Y_train=pd.DataFrame(trainY)
    X_test=pd.DataFrame(testX)
    Y_test=pd.DataFrame(testY)

    sc_X = StandardScaler()
    sc_y = StandardScaler() 
    X= sc_X.fit_transform(X_train)
    y= sc_y.fit_transform(Y_train)
    X1= sc_X.fit_transform(X_test)
    y1= sc_y.fit_transform(Y_test)
    y=y.ravel()
    y1=y1.ravel()


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

    
    print('MAPE',mape)
    print('Classic MAPE', classic_mape)
    print('RMSE',rmse)
    print('MAE',mae)


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
        X_train=pd.DataFrame(trainX)
        Y_train=pd.DataFrame(trainY)
        X_test=pd.DataFrame(testX)
        Y_test=pd.DataFrame(testY)
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X= sc_X.fit_transform(X_train)
        y= sc_y.fit_transform(Y_train)
        X1= sc_X.fit_transform(X_test)
        y1= sc_y.fit_transform(Y_test)
        y=y.ravel()
        y1=y1.ravel()  

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
    X_train=pd.DataFrame(trainX)
    Y_train=pd.DataFrame(trainY)
    X_test=pd.DataFrame(testX)
    Y_test=pd.DataFrame(testY)

    sc_X = StandardScaler()
    sc_y = StandardScaler() 
    X= sc_X.fit_transform(X_train)
    y= sc_y.fit_transform(Y_train)
    X1= sc_X.fit_transform(X_test)
    y1= sc_y.fit_transform(Y_test)
    y=y.ravel()
    y1=y1.ravel()


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

    
    print('MAPE',mape)
    print('Classic MAPE', classic_mape)
    print('RMSE',rmse)
    print('MAE',mae)


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
        X_train=pd.DataFrame(trainX)
        Y_train=pd.DataFrame(trainY)
        X_test=pd.DataFrame(testX)
        Y_test=pd.DataFrame(testY)
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X= sc_X.fit_transform(X_train)
        y= sc_y.fit_transform(Y_train)
        X1= sc_X.fit_transform(X_test)
        y1= sc_y.fit_transform(Y_test)
        y=y.ravel()
        y1=y1.ravel()  

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
    X_train=pd.DataFrame(trainX)
    Y_train=pd.DataFrame(trainY)
    X_test=pd.DataFrame(testX)
    Y_test=pd.DataFrame(testY)

    sc_X = StandardScaler()
    sc_y = StandardScaler() 
    X= sc_X.fit_transform(X_train)
    y= sc_y.fit_transform(Y_train)
    X1= sc_X.fit_transform(X_test)
    y1= sc_y.fit_transform(Y_test)
    y=y.ravel()
    y1=y1.ravel()


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

    
    print('MAPE',mape)
    print('Classic MAPE', classic_mape)
    print('RMSE',rmse)
    print('MAE',mae)

##Proposed Method Hybrid CEEMDAN-EWT LSTM
def proposed_method(
        datasets, look_back, data_partition, cap,
        lstm=True, gru=False, drop_extend=False, bilstm=False,
        bigru=False, trasnformer=False, hilbert=False, save=True
    ):

    # ============================================================
    # 1. DECOMPOSIÇÃO CEEMDAN
    # ============================================================

    s = np.ascontiguousarray(
        np.asarray(datasets, dtype=np.float64).reshape(-1)
    )

    emd_ = CEEMDAN(
        trials=10,
        epsilon=0.05,
        parallel=False
    )

    emd_.noise_seed(12345)

    IMFs = emd_.ceemdan(s)

    print("Shape IMFs:", IMFs.shape, flush=True)

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

    print("Componentes para previsão:", new_ceemdan.shape[1])

    for i in range(new_ceemdan.shape[1]):

        component = new_ceemdan.iloc[:, i].to_numpy()

        print(
            f"Componente {i + 1}: "
            f"mínimo={np.min(component):.8f}, "
            f"máximo={np.max(component):.8f}, "
            f"desvio={np.std(component):.8f}, "
            f"NaN={np.isnan(component).sum()}, "
            f"Inf={np.isinf(component).sum()}",
            flush=True
        )

    # ============================================================
    # 3. PREPARAÇÃO DAS ENTRADAS
    # ============================================================
    # Após a IMF 9 os valores são muito constantes. Dessa forma, não é possível calcular a
    # HHT. Optou-se, neste cenário, por utilizar a própria componente como entrada sem a transformação.
    # Também poderiam ser utilizadas apenas a HHT das componente de 1 a 9.
    if hilbert:

        print("Aplicando HHT...", flush=True)

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

                print(
                    f"Componente {i + 1}: "
                    "HHT não aplicável. Utilizando componente original.",
                    flush=True
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

        print("Utilizando componentes originais como entradas...", flush=True)

        input_components = [
            new_ceemdan.iloc[:, i].to_numpy(
                dtype=np.float32
            ).reshape(-1, 1)
            for i in range(new_ceemdan.shape[1])
        ]

    # ============================================================
    # 4. PREVISÃO INDIVIDUAL DE CADA COMPONENTE
    # ============================================================
    print("Inicio do treinamento do modelo", flush=True)

    pred_test = []
    pred_train = []

    for idx, col in enumerate(new_ceemdan.columns):

        print(f"\nProcessando componente {idx + 1}", flush=True)

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

        trainX, trainY = create_dataset(
            X_train_raw,
            y_train_raw,
            look_back
        )

        testX, testY = create_dataset(
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

        y = np.ascontiguousarray(y.reshape(-1), dtype=np.float32)
        y1 = np.ascontiguousarray(y1.reshape(-1), dtype=np.float32)

        print(
            f"trainX: {trainX.shape} | "
            f"testX: {testX.shape} | "
            f"Features: {n_features}",
            flush=True
        )

        # --------------------------------------------------------
        # Construção da LSTM/GRU
        # --------------------------------------------------------

        K.clear_session()
        gc.collect()

        np.random.seed(1234)
        tf.random.set_seed(1234)

        model = Sequential()

        if gru:
            model.add(
                GRU(
                    units=neuron,
                    input_shape=(trainX.shape[1], trainX.shape[2])
                )
            )

        elif lstm:
            model.add(
                LSTM(
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

    # Soma das previsões de todas as componentes
    result_pred_test = np.vstack(
        [p.reshape(-1) for p in pred_test]
    )

    result_pred_train = np.vstack(
        [p.reshape(-1) for p in pred_train]
    )

    a = np.sum(result_pred_test, axis=0)
    b = np.sum(result_pred_train, axis=0)

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

    print('MAPE', mape)
    print('Classic MAPE', classic_mape)
    print('RMSE', rmse)
    print('MAE', mae)

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

#     print('MAPE', mape)
#     print('RMSE', rmse)
#     print('MAE', mae)

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

#     print('MAPE', mape)
#     print('Classic MAPE', classic_mape)
#     print('RMSE', rmse)
#     print('MAE', mae)

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


#https://emd.readthedocs.io/en/stable/index.html

# Hybrid CEEMDAN-EWT KAN
def proposed_method_with_kan(
    datasets, look_back, data_partition, cap,
    epochs=epochs, batch_size=32, Q=4
):
    import torch
    from torch.utils.data import DataLoader
    import torch
    import torch.nn as nn

    class KANLayer(nn.Module):
        """KAN layer: soma de funções univariadas aprendidas"""
        def __init__(self, in_features, out_features, Q=4):
            super().__init__()
            self.Q = Q
            self.in_features = in_features
            self.out_features = out_features

            # Cada função é uma MLP simples
            self.functions = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.Tanh()
                )
                for _ in range(Q)
            ])

        def forward(self, x):
            # x: (batch, in_features)
            out = 0.0
            for f in self.functions:
                out = out + f(x)
            return out

    class TKANModel(nn.Module):
        def __init__(self, input_size=1, hidden_size=32, output_size=1, Q=4):
            super().__init__()

            self.hidden_size = hidden_size

            # KAN aplicado a cada passo temporal
            self.kan = KANLayer(
                in_features=input_size,
                out_features=hidden_size,
                Q=Q
            )

            # Agregação temporal
            self.temporal_pool = nn.AdaptiveAvgPool1d(1)

            # Saída
            self.fc_out = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            """
            x: (batch, window, input_size)
            """
            B, T, D = x.shape

            # Aplica KAN passo a passo
            x = x.view(B * T, D)
            h = self.kan(x)              # (B*T, hidden)
            h = h.view(B, T, -1)         # (B, T, hidden)

            # Pool temporal
            h = h.permute(0, 2, 1)       # (B, hidden, T)
            h = self.temporal_pool(h)    # (B, hidden, 1)
            h = h.squeeze(-1)            # (B, hidden)

            # Saída
            y = self.fc_out(h)           # (B, output)
            return y


        # Prepara o sinal para o CEEMDAN
    s = np.ascontiguousarray(
        np.asarray(datasets, dtype=np.float64).reshape(-1)
    )

    emd = CEEMDAN(
        trials=10,
        epsilon=0.05,
        parallel=False
    )

    emd.noise_seed(12345)

    IMFs = emd.ceemdan(s)

    print("Shape IMFs:", IMFs.shape, flush=True)

    full_imf = pd.DataFrame(IMFs)
    print("Quantidade IMFs", full_imf.shape[0], flush=True)

    ceemdan1=full_imf.T
    
    imf1=ceemdan1.iloc[:,0]

    ewt,  mfb ,boundaries = ewtpy.EWT1D(imf1, N =3)

    df_ewt=pd.DataFrame(ewt)

    df_ewt.drop(df_ewt.columns[2],axis=1,inplace=True)
    denoised=df_ewt.sum(axis = 1, skipna = True) 
    ceemdan_without_imf1=ceemdan1.iloc[:,1:]
    new_ceemdan=pd.concat([denoised,ceemdan_without_imf1],axis=1)    
    
    pred_test=[]
    test_ori=[]
    pred_train=[]
    train_ori=[]

    for col in new_ceemdan:

        series = new_ceemdan[col].values.reshape(-1, 1)

        # Dataset janela temporal
        X, y = create_dataset(series, look_back)

        split = int(len(X) * data_partition)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Torch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
        X_test  = torch.tensor(X_test,  dtype=torch.float32).unsqueeze(-1)

        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        y_test_torch = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        train_ds = torch.utils.data.TensorDataset(X_train, y_train)
        test_ds  = torch.utils.data.TensorDataset(X_test, y_test_torch)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        def train_model(model, train_loader, val_loader=None, epochs=50, lr=1e-3, device=None):
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            train_losses = []
            val_losses = []

            for epoch in range(epochs):
                model.train()
                epoch_loss = 0.0

                for xb, yb in train_loader:
                    xb = xb.to(device).float()
                    yb = yb.to(device).float()

                    optimizer.zero_grad()
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                epoch_loss /= len(train_loader)
                train_losses.append(epoch_loss)

                # Validação (opcional)
                if val_loader is not None:
                    model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for xb, yb in val_loader:
                            xb = xb.to(device).float()
                            yb = yb.to(device).float()
                            pred = model(xb)
                            loss = criterion(pred, yb)
                            val_loss += loss.item()

                    val_loss /= len(val_loader)
                    val_losses.append(val_loss)

                    print(
                        f"Epoch [{epoch+1}/{epochs}] "
                        f"Train MSE: {epoch_loss:.6f} | Val MSE: {val_loss:.6f}"
                    )
                else:
                    print(
                        f"Epoch [{epoch+1}/{epochs}] "
                        f"Train MSE: {epoch_loss:.6f}"
                    )

            return train_losses, val_losses


        # -----------------------------
        # TKAN REAL (o seu)
        # -----------------------------
        model = TKANModel(
            input_size=1,
            hidden_size=32,
            output_size=1,
            Q=Q
        )

        train_model(
            model,
            train_loader,
            val_loader=None,
            epochs=epochs
        )

        # -----------------------------
        # Previsão
        # -----------------------------
        model.eval()
        preds = []

        with torch.no_grad():
            for xb, _ in test_loader:
                pred = model(xb)
                preds.append(pred.cpu().numpy())

        preds = np.concatenate(preds).reshape(-1, 1)
        pred_test.append(preds)

    # -----------------------------
    # Reconstrução do sinal
    # -----------------------------
    result_pred_test = pd.DataFrame.from_records(pred_test)
    a = result_pred_test.sum(axis=0).values.reshape(-1, 1)

    # -----------------------------
    # Métricas
    # -----------------------------
    mape = np.mean(np.abs((y_test - a)) / cap) * 100
    rmse = sqrt(mean_squared_error(y_test, a))
    mae = metrics.mean_absolute_error(y_test, a)

    print("MAPE:", mape)
    print("RMSE:", rmse)
    print("MAE:", mae)

    # Save the real and predict values
    np.savetxt(os.path.join(BASE_DIR, 'y_test.txt'), y_test.values.flatten(), fmt='%.6f')
    np.savetxt(os.path.join(BASE_DIR, 'y_pred_kan.txt'), a.values.flatten(), fmt='%.6f')

    # dates = data1['Month'].iloc[-len(y_test):]
    # dates = pd.to_datetime(dates)

    # np.savetxt(os.path.join(BASE_DIR, 'dates.txt'), dates.astype(str), fmt='%s')

    return  y_test.values.flatten(), a.values.flatten(), dates.astype(str).values


def proposed_method_with_deeponet(
    datasets, look_back, data_partition, cap,
    epochs=50, batch_size=32, hidden_dim=64
):
    import torch
    from torch.utils.data import DataLoader

    class DeepONetDataset(torch.utils.data.Dataset):
        def __init__(self, X_branch, X_trunk, y):
            self.Xb = torch.tensor(X_branch, dtype=torch.float32)
            self.Xt = torch.tensor(X_trunk, dtype=torch.float32)
            self.y  = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.Xb[idx], self.Xt[idx], self.y[idx]
    
    import torch
    import torch.nn as nn

    class BranchNet(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

        def forward(self, x):
            return self.net(x)


    class TrunkNet(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

        def forward(self, x):
            return self.net(x)


    class DeepONet(nn.Module):
        def __init__(self, branch_input_dim, trunk_input_dim, hidden_dim, output_dim=1):
            super().__init__()
            self.branch = BranchNet(branch_input_dim, hidden_dim)
            self.trunk  = TrunkNet(trunk_input_dim, hidden_dim)
            self.fc_out = nn.Linear(hidden_dim, output_dim)

        def forward(self, xb, xt):
            # xb: (batch, look_back)
            # xt: (batch, trunk_dim)
            b = self.branch(xb)
            t = self.trunk(xt)
            y = self.fc_out(b * t)
            return y

    # Prepara o sinal para o CEEMDAN
    s = np.ascontiguousarray(
        np.asarray(datasets, dtype=np.float64).reshape(-1)
    )

    emd = CEEMDAN(
        trials=10,
        epsilon=0.05,
        parallel=False
    )

    emd.noise_seed(12345)

    IMFs = emd.ceemdan(s)

    print("Shape IMFs:", IMFs.shape, flush=True)

    full_imf = pd.DataFrame(IMFs)
    print("Quantidade IMFs", full_imf.shape[0], flush=True)

    ceemdan1=full_imf.T
    
    imf1=ceemdan1.iloc[:,0]

    ewt,  mfb ,boundaries = ewtpy.EWT1D(imf1, N =3)

    df_ewt=pd.DataFrame(ewt)

    df_ewt.drop(df_ewt.columns[2],axis=1,inplace=True)
    denoised=df_ewt.sum(axis = 1, skipna = True) 
    ceemdan_without_imf1=ceemdan1.iloc[:,1:]
    new_ceemdan=pd.concat([denoised,ceemdan_without_imf1],axis=1)    
    
    pred_test=[]
    test_ori=[]
    pred_train=[]
    train_ori=[]

    # -----------------------------
    # Loop por IMF
    # -----------------------------
    for col in new_ceemdan.columns:

        series = new_ceemdan[col].values.reshape(-1, 1)
        X, y = create_dataset(series, look_back)

        split = int(len(X) * data_partition)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Branch = janela temporal
        Xb_train = X_train
        Xb_test  = X_test

        # Trunk = tempo normalizado
        Xt_train = np.linspace(0, 1, len(Xb_train)).reshape(-1, 1)
        Xt_test  = np.linspace(0, 1, len(Xb_test)).reshape(-1, 1)

        train_ds = DeepONetDataset(Xb_train, Xt_train, y_train)
        test_ds  = DeepONetDataset(Xb_test, Xt_test, y_test)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # -----------------------------
        # Modelo DeepONet
        # -----------------------------
        model = DeepONet(
            branch_input_dim=look_back,
            trunk_input_dim=1,
            hidden_dim=hidden_dim,
            output_dim=1
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        criterion = nn.MSELoss()

        # -----------------------------
        # Treinamento
        # -----------------------------
        for epoch in range(epochs):
            model.train()
            loss_epoch = 0.0

            for xb, xt, yb in train_loader:
                xb, xt, yb = xb.to(device), xt.to(device), yb.to(device)

                optimizer.zero_grad()
                pred = model(xb, xt)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - MSE: {loss_epoch/len(train_loader):.6f}")

        # -----------------------------
        # Previsão
        # -----------------------------
        model.eval()
        preds = []

        with torch.no_grad():
            for xb, xt, _ in test_loader:
                xb, xt = xb.to(device), xt.to(device)
                preds.append(model(xb, xt).cpu().numpy())

        pred_test.append(np.concatenate(preds).reshape(-1, 1))

    # -----------------------------
    # Reconstrução do sinal
    # -----------------------------
    a = np.sum(np.array(pred_test), axis=0)

    # dates = data1['Month'].iloc[-len(y_test):]
    # dates = pd.to_datetime(dates)

    # -----------------------------
    # Métricas
    # -----------------------------
    mape = np.mean(np.abs(y_test - a) / cap) * 100
    rmse = sqrt(mean_squared_error(y_test, a))
    mae  = metrics.mean_absolute_error(y_test, a)

    print("MAPE:", mape)
    print("RMSE:", rmse)
    print("MAE:", mae)

    return y_test.values.flatten(), a.values.flatten()


def proposed_method_with_lstm_deeponet(
    datasets, look_back, data_partition, cap,
    epochs=50, batch_size=32, hidden_dim=64
):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================================
    # Dataset
    # ==========================================================
    class HybridDataset(Dataset):
        def __init__(self, X_seq, X_time, y):
            self.Xs = torch.tensor(X_seq, dtype=torch.float32)
            self.Xt = torch.tensor(X_time, dtype=torch.float32)
            self.y  = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.Xs[idx], self.Xt[idx], self.y[idx]

    # ==========================================================
    # Branch: LSTM
    # ==========================================================
    class LSTMBranch(nn.Module):
        def __init__(self, input_dim=1, hidden_dim=64):
            super().__init__()
            self.lstm = nn.LSTM(
                input_dim,
                hidden_dim,
                batch_first=True
            )

        def forward(self, x):
            # x: (batch, look_back, 1)
            _, (h_n, _) = self.lstm(x)
            return h_n[-1]  # (batch, hidden_dim)

    # ==========================================================
    # Trunk
    # ==========================================================
    class TrunkNet(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

        def forward(self, x):
            return self.net(x)

    # ==========================================================
    # LSTM–DeepONet
    # ==========================================================
    class LSTMDeepONet(nn.Module):
        def __init__(self, trunk_dim, hidden_dim):
            super().__init__()
            self.branch = LSTMBranch(1, hidden_dim)
            self.trunk  = TrunkNet(trunk_dim, hidden_dim)

        def forward(self, x_seq, x_time):
            b = self.branch(x_seq)
            t = self.trunk(x_time)
            y = torch.sum(b * t, dim=1, keepdim=True)
            return y

        # Prepara o sinal para o CEEMDAN
    s = np.ascontiguousarray(
        np.asarray(datasets, dtype=np.float64).reshape(-1)
    )

    emd = CEEMDAN(
        trials=10,
        epsilon=0.05,
        parallel=False
    )

    emd.noise_seed(12345)

    IMFs = emd.ceemdan(s)

    print("Shape IMFs:", IMFs.shape, flush=True)

    full_imf = pd.DataFrame(IMFs)
    print("Quantidade IMFs", full_imf.shape[0], flush=True)

    ceemdan1=full_imf.T
    
    imf1=ceemdan1.iloc[:,0]

    ewt,  mfb ,boundaries = ewtpy.EWT1D(imf1, N =3)

    df_ewt=pd.DataFrame(ewt)

    df_ewt.drop(df_ewt.columns[2],axis=1,inplace=True)
    denoised=df_ewt.sum(axis = 1, skipna = True) 
    ceemdan_without_imf1=ceemdan1.iloc[:,1:]
    new_ceemdan=pd.concat([denoised,ceemdan_without_imf1],axis=1)    
    
    pred_test=[]
    test_ori=[]
    pred_train=[]
    train_ori=[]

    # ==========================================================
    # Loop por IMF
    # ==========================================================
    for col in new_ceemdan.columns:

        series = new_ceemdan[col].values.reshape(-1, 1)
        X, y = create_dataset(series, look_back)

        split = int(len(X) * data_partition)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # ---- Branch input (LSTM)
        Xb_train = X_train.reshape(-1, look_back, 1)
        Xb_test  = X_test.reshape(-1, look_back, 1)

        # ---- Trunk input (tempo enriquecido)
        t_train = np.linspace(0, 1, len(Xb_train)).reshape(-1, 1)
        t_test  = np.linspace(0, 1, len(Xb_test)).reshape(-1, 1)

        Xt_train = np.hstack([
            t_train,
            np.sin(2 * np.pi * t_train),
            np.cos(2 * np.pi * t_train)
        ])

        Xt_test = np.hstack([
            t_test,
            np.sin(2 * np.pi * t_test),
            np.cos(2 * np.pi * t_test)
        ])

        train_ds = HybridDataset(Xb_train, Xt_train, y_train)
        test_ds  = HybridDataset(Xb_test, Xt_test, y_test)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        model = LSTMDeepONet(trunk_dim=3, hidden_dim=hidden_dim).to(device)

        criterion = nn.MSELoss()

        # ==========================================================
        # Treinamento
        # ==========================================================
        for epoch in range(epochs):
            model.train()
            loss_epoch = 0.0

            for xb, xt, yb in train_loader:
                xb, xt, yb = xb.to(device), xt.to(device), yb.to(device)

                optimizer.zero_grad()
                pred = model(xb, xt)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"IMF {col} | Epoch {epoch+1}/{epochs} | MSE {loss_epoch/len(train_loader):.6f}")

        # ==========================================================
        # Previsão
        # ==========================================================
        model.eval()
        preds = []

        with torch.no_grad():
            for xb, xt, _ in test_loader:
                xb, xt = xb.to(device), xt.to(device)
                preds.append(model(xb, xt).cpu().numpy())

        pred_test.append(np.concatenate(preds).reshape(-1, 1))

    # ==========================================================
    # Reconstrução do sinal
    # ==========================================================
    a = np.sum(np.array(pred_test), axis=0)
    
    # dates = data1['Month'].iloc[-len(y_test):]
    # dates = pd.to_datetime(dates)

    # ==========================================================
    # Métricas
    # ==========================================================
    mape = np.mean(np.abs(y_test - a) / cap) * 100
    rmse = sqrt(mean_squared_error(y_test, a))
    mae  = metrics.mean_absolute_error(y_test, a)

    print("MAPE:", mape)
    print("RMSE:", rmse)
    print("MAE:", mae)

    return y_test.values.flatten(), a.values.flatten()


if __name__=='__main__':
    df = pd.read_csv('dataset/final_la_haute_R0711.csv')
    df['Date'] = pd.to_datetime(df['Date_time'], dayfirst=True)
    df['Year'] = df['Date'].dt.year 
    df['Month'] = df['Date'].dt.month 
    new_data=df[['Month','Year','Date','P_avg']]
    new_data=new_data[new_data.Year == 2017]

    cap=max(new_data['P_avg'])
    i=[1] # enter month value, i.e January = 1
    look_back=6
    data_partition=0.8
    proposed_method(new_data,i,look_back,data_partition,cap)