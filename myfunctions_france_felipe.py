#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
from PyEMD import CEEMDAN
import math
import tensorflow as tf
import numpy
# Ajusta configurações do tensorflow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["PYTHONHASHSEED"] = "1234"
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import backend as K
from sklearn import metrics
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

epochs=10

#convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

import tensorflow as tf

def run_pipeline(datasets, data1, look_back, model_func,
                 use_cv=False, n_splits=3, data_partition=0.7, cap=None):

    # =========================
    # TREINO PADRÃO
    # =========================
    def run_model(train, test):

        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)

        # manter exatamente como no código antigo
        X_train = pd.DataFrame(trainX)
        Y_train = pd.DataFrame(trainY)
        X_test  = pd.DataFrame(testX)
        Y_test  = pd.DataFrame(testY)

        sc_X = StandardScaler()
        sc_y = StandardScaler()

        # igual ao antigo (fit_transform em ambos)
        X  = sc_X.fit_transform(X_train)
        X1 = sc_X.fit_transform(X_test)

        y  = sc_y.fit_transform(Y_train).ravel()
        y1 = sc_y.fit_transform(Y_test).ravel()

        # modelo
        y_pred_scaled = model_func(X, y, X1)

        # inverter escala
        y_pred = sc_y.inverse_transform(
            np.asarray(y_pred_scaled).reshape(-1,1)
        ).ravel()

        y_test = sc_y.inverse_transform(
            np.asarray(y1).reshape(-1,1)
        ).ravel()

        # métricas
        mape = np.mean(np.abs(y_test - y_pred) / cap) * 100
        classic_mape = np.mean(np.abs((y_test - a) / y_test)) * 100
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        mae  = np.mean(np.abs(y_test - y_pred))

        return mape, classic_mape, rmse, mae, y_test, y_pred

    # =========================
    # CROSS VALIDATION
    # =========================
    if use_cv:

        tscv = TimeSeriesSplit(n_splits=n_splits)

        y_test_all, y_pred_all, dates_all = [], [], []
        mape_list, classic_map_list, rmse_list, mae_list = [], [], [], []

        for train_index, test_index in tscv.split(datasets):

            train = datasets[train_index]
            test = datasets[test_index]

            mape, classic_mape, rmse, mae, y_test, y_pred = run_model(train, test)

            mape_list.append(mape)
            classic_map_list.append(classic_mape)
            rmse_list.append(rmse)
            mae_list.append(mae)

            y_test_all.extend(np.asarray(y_test).ravel())
            y_pred_all.extend(np.asarray(y_pred).ravel())

            valid_idx = test_index[look_back+1:]
            valid_idx = valid_idx[:len(y_test)]

            dates_fold = pd.to_datetime(data1['Date'].iloc[valid_idx])
            dates_all.extend(np.asarray(dates_fold).astype(str))

        print("MAPE", np.mean(mape_list))
        print("Classic_MAPE", np.mean(classic_map_list))
        print("RMSE", np.mean(rmse_list))
        print("MAE", np.mean(mae_list))

        return (
            np.asarray(y_test_all),
            np.asarray(y_pred_all),
            np.asarray(dates_all)
        )

    # =========================
    # HOLD OUT
    # =========================
    train_size = int(len(datasets) * data_partition)

    train = datasets[:train_size]
    test = datasets[train_size:]

    mape, classic_mape, rmse, mae, y_test, y_pred = run_model(train, test)

    print("MAPE", mape)
    print("Classic_map", classic_mape)
    print("RMSE", rmse)
    print("MAE", mae)

    dates = pd.to_datetime(data1['Date'].iloc[
        train_size + look_back:
        train_size + look_back + len(y_test)
    ])

    return (
        np.asarray(y_test),
        np.asarray(y_pred),
        np.asarray(dates).astype(str)
    )

# =============================================================================
# DIRECT MODELS
# =============================================================================

def svr_model(new_data, i, look_back, data_partition, cap,
              use_cv=False, n_splits=3):

    import numpy as np
    from sklearn.svm import SVR

    x = i
    data1 = new_data.loc[new_data['Month'].isin(x)]
    data1 = data1.reset_index(drop=True).dropna()

    datasets = data1['P_avg'].values.reshape(-1, 1)

    # =========================
    # modelo plugável
    # =========================
    def model_func(X_train, y_train, X_test):

        model = SVR(kernel='rbf')
        model.fit(X_train, y_train)

        return model.predict(X_test)

    y_test, y_pred, dates = run_pipeline(
        datasets=datasets,
        data1=data1,
        look_back=look_back,
        model_func=model_func,
        use_cv=use_cv,
        n_splits=n_splits,
        data_partition=data_partition,
        cap=cap
    )

    suffix = "_cv" if use_cv else ""

    np.savetxt(f'y_test{suffix}.txt', y_test, fmt='%.6f')
    np.savetxt(f'y_pred_svr_model{suffix}.txt', y_pred, fmt='%.6f')
    np.savetxt(f'dates{suffix}.txt', dates, fmt='%s')

    return y_test, y_pred, dates

def ann_model(new_data, i, look_back, data_partition, cap,
              use_cv=False, n_splits=3):

    x = i
    data1 = new_data.loc[new_data['Month'].isin(x)]
    data1 = data1.reset_index(drop=True).dropna()

    datasets = data1['P_avg'].values.reshape(-1,1)

    # =========================
    # modelo ANN plugável
    # =========================
    def model_func(X_train, y_train, X_test):

        X_train = np.asarray(X_train)
        X_test = np.asarray(X_test)

        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test  = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

        model = Sequential()
        model.add(Dense(128, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(1))

        model.compile(loss='mse', optimizer='adam')

        model.fit(X_train, y_train, verbose=0)

        y_pred = model.predict(X_test, verbose=0)

        return y_pred.ravel()

    y_test, y_pred, dates = run_pipeline(
        datasets=datasets,
        data1=data1,
        look_back=look_back,
        model_func=model_func,
        use_cv=use_cv,
        n_splits=n_splits,
        data_partition=data_partition,
        cap=cap
    )

    suffix = "_cv" if use_cv else ""

    np.savetxt(f'y_test{suffix}.txt', y_test, fmt='%.6f')
    np.savetxt(f'y_pred_ann_model{suffix}.txt', y_pred, fmt='%.6f')
    np.savetxt(f'dates{suffix}.txt', dates, fmt='%s')

    return y_test, y_pred, dates


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
    import tensorflow as tf

    import numpy
    
    numpy.random.seed(1234)
    tf.random.set_seed(1234)

    
    from sklearn.ensemble import RandomForestRegressor
    

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
    mape=numpy.mean((numpy.abs(y_test-y_pred_test1_rf))/cap)*100
    classic_mape = np.mean(np.abs((y_test - a) / y_test)) * 100
    rmse= sqrt(mean_squared_error(y_test,y_pred_test1_rf))
    mae=metrics.mean_absolute_error(y_test,y_pred_test1_rf)

    print('MAPE',mape)
    print('Classic MAPE', classic_mape)
    print('RMSE',rmse)
    print('MAE',mae)


# In[23]:


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
    import tensorflow as tf

    
    import numpy
    numpy.random.seed(1234)
    tf.random.set_seed(1234)


    trainX1 = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX1 = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))
    
    
    numpy.random.seed(1234)
    import tensorflow as tf
    tf.random.set_seed(1234)
    
    
    import os 
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Activation
    from tensorflow.keras.layers import LSTM


    neuron=128
    model = Sequential()
    model.add(LSTM(units = neuron,input_shape=(trainX1.shape[1], trainX1.shape[2])))
    model.add(Dense(1))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mse',optimizer=optimizer)

    model.fit(trainX1, y, epochs = epochs, batch_size = 64,verbose=0)
  # make predictions
    y_pred_train = model.predict(trainX1)
    y_pred_test = model.predict(testX1)
    y_pred_test= numpy.array(y_pred_test).ravel()

    y_pred_test=pd.DataFrame(y_pred_test)
    y_pred_test1= sc_y.inverse_transform (y_pred_test)
    y1=pd.DataFrame(y1)
      
    y_test= sc_y.inverse_transform (y1)
       
    #summarize the fit of the model
    mape=numpy.mean((numpy.abs(y_test-y_pred_test1))/cap)*100
    classic_mape = np.mean(np.abs((y_test - a) / y_test)) * 100
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

    from PyEMD import EMD
    import ewtpy
    
    emd = EMD()

    IMFs = emd(s)

    full_imf=pd.DataFrame(IMFs)
    data_decomp=full_imf.T

    pred_test=[]
    test_ori=[]
    pred_train=[]
    train_ori=[]
    batch_size=64
    neuron=128
    lr=0.001
    optimizer='Adam'

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

        import numpy
        trainX = numpy.reshape(X, (X.shape[0],X.shape[1],1))
        testX = numpy.reshape(X1, (X1.shape[0],X1.shape[1],1))
    
        

        numpy.random.seed(1234)
        import tensorflow as tf
        tf.random.set_seed(1234)

    
        import os 
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, Activation
        from tensorflow.keras.layers import LSTM


        neuron=128
        model = Sequential()
        model.add(LSTM(units = neuron,input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
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
    mape=numpy.mean((numpy.abs(y_test-a))/cap)*100
    classic_mape = np.mean(np.abs((y_test - a) / y_test)) * 100
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

    from PyEMD import EEMD
    import ewtpy
    
    emd = EEMD(noise_width=0.02)
    emd.noise_seed(12345)

    IMFs = emd(s)

    full_imf=pd.DataFrame(IMFs)
    data_decomp=full_imf.T
    


    pred_test=[]
    test_ori=[]
    pred_train=[]
    train_ori=[]

    batch_size=64
    neuron=128
    lr=0.001
    optimizer='Adam'

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

        import numpy
        trainX = numpy.reshape(X, (X.shape[0],X.shape[1],1))
        testX = numpy.reshape(X1, (X1.shape[0],X1.shape[1],1))
    
        

        numpy.random.seed(1234)
        import tensorflow as tf
        tf.random.set_seed(1234)

    
        import os 
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, Activation
        from tensorflow.keras.layers import LSTM


        neuron=128
        model = Sequential()
        model.add(LSTM(units = neuron,input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
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
    mape=numpy.mean((numpy.abs(y_test-a))/cap)*100
    classic_mape = np.mean(np.abs((y_test - a) / y_test)) * 100
    rmse= sqrt(mean_squared_error(y_test,a))
    mae=metrics.mean_absolute_error(y_test,a)

    
    print('MAPE',mape)
    print('Classic MAPE', classic_mape)
    print('RMSE',rmse)
    print('MAE',mae)


# In[26]:


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

    from PyEMD import CEEMDAN
    
    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)

    IMFs = emd(s)

    full_imf=pd.DataFrame(IMFs)
    data_decomp=full_imf.T
    


    pred_test=[]
    test_ori=[]
    pred_train=[]
    train_ori=[]

    batch_size=64
    neuron=128
    lr=0.001
    optimizer='Adam'

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

        import numpy
        trainX = numpy.reshape(X, (X.shape[0],X.shape[1],1))
        testX = numpy.reshape(X1, (X1.shape[0],X1.shape[1],1))
    
        

        numpy.random.seed(1234)
        import tensorflow as tf
        tf.random.set_seed(1234)

    
        import os 
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, Activation
        from tensorflow.keras.layers import LSTM


        neuron=128
        model = Sequential()
        model.add(LSTM(units = neuron,input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
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
    mape=numpy.mean((numpy.abs(y_test-a))/cap)*100
    classic_mape = np.mean(np.abs((y_test - a) / y_test)) * 100
    rmse= sqrt(mean_squared_error(y_test,a))
    mae=metrics.mean_absolute_error(y_test,a)

    
    print('MAPE',mape)
    print('Classic MAPE', classic_mape)
    print('RMSE',rmse)
    print('MAE',mae)


def run_cv_hybrid_models(method,
                 new_data, i,
                 look_back, cap,
                 n_splits=3):

    import numpy as np
    from sklearn.model_selection import TimeSeriesSplit

    data1 = new_data.loc[new_data['Month'].isin(i)]
    data1 = data1.reset_index(drop=True).dropna()

    tscv = TimeSeriesSplit(n_splits=n_splits)

    mape_list, classic_mape_list, rmse_list, mae_list = [], [], [], []
    y_test_all, y_pred_all, dates_all = [], [], []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(data1)):

        print(f"\nFold {fold+1} - {method.__name__}")

        data_fold = data1.iloc[
            np.concatenate([train_idx, test_idx])
        ].reset_index(drop=True).copy()

        split_ratio = len(train_idx) / len(data_fold)

        y_test, y_pred, dates = method(
            new_data=data_fold,
            i=data_fold['Month'],
            look_back=look_back,
            data_partition=split_ratio,
            cap=cap
        )

        mape = np.mean(np.abs(y_test - y_pred) / cap) * 100
        classic_mape = np.mean(np.abs((y_test - a) / y_test)) * 100
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        mae  = np.mean(np.abs(y_test - y_pred))

        mape_list.append(mape)
        classic_mape_list.append(classic_mape)
        rmse_list.append(rmse)
        mae_list.append(mae)

        y_test_all.extend(y_test)
        y_pred_all.extend(y_pred)
        dates_all.extend(dates)

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


##Proposed Method Hybrid CEEMDAN-EWT LSTM
def proposed_method(new_data,i,look_back,data_partition,cap, save=True):

    x=i
    data1=new_data.loc[new_data['Month'].isin(x)]
    data1=data1.reset_index(drop=True)
    data1=data1.dropna()
    
    datas=data1['P_avg']
    datas_wind=pd.DataFrame(datas)
    dfs=datas
    s = dfs.values

    from PyEMD import EMD,EEMD,CEEMDAN
    import numpy

    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)

    IMFs = emd(s)

    full_imf=pd.DataFrame(IMFs)
    print("Quantidade IMFs", full_imf.shape[0])

    ceemdan1=full_imf.T
    
    imf1=ceemdan1.iloc[:,0]
    imf_dataps=numpy.array(imf1)
    imf_datasetss= imf_dataps.reshape(-1,1)
    imf_new_datasets=pd.DataFrame(imf_datasetss)

    import ewtpy

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

    batch_size=64
    lr=0.001
    optimizer='Adam'

    for col in new_ceemdan:

        datasetss2=pd.DataFrame(new_ceemdan[col])
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

        import numpy

        trainX = numpy.reshape(X, (X.shape[0], X.shape[1],1))
        testX = numpy.reshape(X1, (X1.shape[0], X1.shape[1],1))

        numpy.random.seed(1234)
        import tensorflow as tf
        tf.random.set_seed(1234)

        
        import os 
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, Activation
        from tensorflow.keras.layers import LSTM


        # LSTM Archithecture Summary:
        # - input_shape= (qtd. registros/janela temporal, características (1 neste caso)).
        # - Camada Sequencial = 128 neurônios. ([xᵢ,t-5] → [xᵢ,t-4] → [xᵢ,t-3] → [xᵢ,t-2] → [xᵢ,t-1] → [xᵢ,t] - 128 Unidades)
        # - Camada Densa = 1 neurônio. Previsao. Input: ht (último estado oculto) pertencente a R128. y^​=w'hT​+b. output_shape=(qtd. registros, 1).
        neuron=128
        model = Sequential()
        model.add(LSTM(units = neuron,input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mse',optimizer=optimizer)

        model.fit(trainX, y, epochs = epochs, batch_size = batch_size,verbose=0)

         # make predictions
        y_pred_train = model.predict(trainX)
        y_pred_test = model.predict(testX)
        
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

    import numpy

    trainX = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    numpy.random.seed(1234)
    import tensorflow as tf

    y1=pd.DataFrame(y1)
    y=pd.DataFrame(y)

    y_test= sc_y.inverse_transform (y1)
    y_train= sc_y.inverse_transform (y)


    a= pd.DataFrame(a)    
    y_test= pd.DataFrame(y_test)    


    #summarize the fit of the model
    mape=numpy.mean((numpy.abs(y_test-a))/cap)*100
    classic_mape = np.mean(np.abs((y_test - a) / y_test)) * 100
    rmse= sqrt(mean_squared_error(y_test,a))
    mae=metrics.mean_absolute_error(y_test,a)

    
    print('MAPE',mape)
    print('Classic MAPE', classic_mape)
    print('RMSE',rmse)
    print('MAE',mae)

    dates = pd.to_datetime(data1['Date'].iloc[
        train_size + look_back:
        train_size + look_back + len(y_test)
    ])

    if save:

        # Save the real and predict values
        np.savetxt(os.path.join(BASE_DIR, 'y_test.txt'), y_test.values.flatten(), fmt='%.6f')
        np.savetxt(os.path.join(BASE_DIR, 'y_pred_proposed_method.txt'), a.values.flatten(), fmt='%.6f')

        np.savetxt(os.path.join(BASE_DIR, 'dates.txt'), dates.astype(str), fmt='%s')

    return y_test.values.flatten(), a.values.flatten(), dates.astype(str)


# Proposed Method Hybrid CEEMDAN (without EWT) with Hilbert transform. This is a variation
# of the Hilbert-Huang Transformer (HHT)

# def proposed_method_hilbert_transform(new_data, i, look_back, data_partition, cap):

#     import numpy as np
#     import pandas as pd
#     from math import sqrt
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.metrics import mean_squared_error
#     from sklearn import metrics
#     from scipy.signal import hilbert
#     from PyEMD import CEEMDAN
#     import tensorflow as tf
#     from tensorflow.keras.models import Sequential
#     from tensorflow.keras.layers import Dense, LSTM
#     from tensorflow.keras import backend as K
#     import gc
#     import os

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

#     batch_size = 64

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

#         model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.001))

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

def proposed_method_hilbert_transform(new_data, i, look_back, data_partition, cap):

    import numpy as np
    import pandas as pd
    from math import sqrt
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    from sklearn import metrics
    from scipy.signal import hilbert
    from PyEMD import CEEMDAN
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM
    import os

    # =========================
    # 1. Data
    # =========================
    data1 = new_data.loc[new_data['Month'].isin(i)]
    data1 = data1.reset_index(drop=True).dropna()

    y_series = data1['P_avg'].values.reshape(-1, 1)

    # =========================
    # 2. CEEMDAN
    # =========================
    ceemdan = CEEMDAN(epsilon=0.05)
    ceemdan.noise_seed(12345)

    IMFs = ceemdan(y_series.flatten())
    ceemdan_df = pd.DataFrame(IMFs).T

    # Limitar IMFs (controle de custo)
    # MAX_IMFS = 5
    # ceemdan_df = ceemdan_df.iloc[:, :MAX_IMFS]

    imfs_array = ceemdan_df.values

    # =========================
    # 3. Hilbert Transform
    # =========================
    analytic = hilbert(imfs_array, axis=0)

    amplitude = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic), axis=0)

    inst_freq = np.diff(phase, axis=0)
    inst_freq = np.vstack([inst_freq, np.zeros((1, inst_freq.shape[1]))])

    # =========================
    # 4. Features finais
    # =========================
    features = np.concatenate([
        imfs_array,
        amplitude,
        inst_freq
    ], axis=1)

    # =========================
    # 5. Dataset supervisionado
    # =========================
    def create_multivariate_dataset(X, y, look_back):
        Xs, ys = [], []
        for i in range(len(X) - look_back):
            Xs.append(X[i:i+look_back, :])
            ys.append(y[i+look_back])
        return np.array(Xs), np.array(ys)

    X_all, y_all = create_multivariate_dataset(features, y_series, look_back)

    # =========================
    # 6. Split
    # =========================
    train_size = int(len(X_all) * data_partition)

    X_train, X_test = X_all[:train_size], X_all[train_size:]
    y_train, y_test = y_all[:train_size], y_all[train_size:]

    # =========================
    # 7. Scaling
    # =========================
    sc_X = StandardScaler()
    sc_y = StandardScaler()

    # reshape para scaler
    X_train_2d = X_train.reshape(-1, X_train.shape[2])
    X_test_2d  = X_test.reshape(-1, X_test.shape[2])

    X_train_scaled = sc_X.fit_transform(X_train_2d)
    X_test_scaled  = sc_X.transform(X_test_2d)

    # volta para 3D
    X_train = X_train_scaled.reshape(X_train.shape)
    X_test  = X_test_scaled.reshape(X_test.shape)

    y_train = sc_y.fit_transform(y_train)
    y_test_scaled = sc_y.transform(y_test)

    # =========================
    # 8. Modelo LSTM
    # =========================
    np.random.seed(1234)
    tf.random.set_seed(1234)

    model = Sequential()
    model.add(LSTM(128, input_shape=(look_back, X_train.shape[2])))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.001))

    model.fit(X_train, y_train,
              epochs=epochs,
              batch_size=32,
              verbose=1)

    # =========================
    # 9. Previsão
    # =========================
    y_pred = model.predict(X_test)

    # voltar escala
    y_pred = sc_y.inverse_transform(y_pred)
    y_test = sc_y.inverse_transform(y_test_scaled)

    dates = data1['Month'].iloc[-len(y_test):]
    dates = pd.to_datetime(dates)

    # =========================
    # 10. Métricas
    # =========================
    mape = np.mean(np.abs(y_test - y_pred) / cap) * 100
    classic_mape = np.mean(np.abs((y_test - a) / y_test)) * 100
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    mae = metrics.mean_absolute_error(y_test, y_pred)

    print('MAPE', mape)
    print('RMSE', rmse)
    print('MAE', mae)

    # Save metrics
    try:
        BASE_DIR = "/app"
        np.savetxt(os.path.join(BASE_DIR, 'y_test_hht.txt'),
                   y_test.flatten(), fmt='%.6f')
        np.savetxt(os.path.join(BASE_DIR, 'y_pred_hht.txt'),
                   y_pred.flatten(), fmt='%.6f')
    except:
        pass

    return y_test.flatten(), y_pred.flatten(), dates.astype(str)


##Proposed Method Hybrid CEEMDAN-EWT LSTM with Stable Layer

def proposed_method_stable_layer(new_data,i,look_back,data_partition,cap):

    x=i
    data1=new_data.loc[new_data['Month'].isin(x)]
    data1=data1.reset_index(drop=True)
    data1=data1.dropna()
    
    datas=data1['P_avg']
    datas_wind=pd.DataFrame(datas)
    dfs=datas
    s = dfs.values

    from PyEMD import EMD,EEMD,CEEMDAN
    import numpy

    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)

    IMFs = emd(s)

    full_imf=pd.DataFrame(IMFs)
    ceemdan1=full_imf.T
    
    imf1=ceemdan1.iloc[:,0]
    imf_dataps=numpy.array(imf1)
    imf_datasetss= imf_dataps.reshape(-1,1)
    imf_new_datasets=pd.DataFrame(imf_datasetss)

    import ewtpy

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

    batch_size=64
    lr=0.001
    optimizer='Adam'

    for col in new_ceemdan:

        datasetss2=pd.DataFrame(new_ceemdan[col])
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

        import numpy

        trainX = numpy.reshape(X, (X.shape[0], X.shape[1],1))
        testX = numpy.reshape(X1, (X1.shape[0], X1.shape[1],1))

        numpy.random.seed(1234)
        import tensorflow as tf
        tf.random.set_seed(1234)

        import os 
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.layers import LSTM


        # LSTM Archithecture Summary:
        # - input_shape= (qtd. registros/janela temporal, características (1 neste caso)).
        # - Camada Sequencial = 128 neurônios. ([xᵢ,t-5] → [xᵢ,t-4] → [xᵢ,t-3] → [xᵢ,t-2] → [xᵢ,t-1] → [xᵢ,t] - 128 Unidades)
        # - Camada Densa = 1 neurônio. Previsao. Input: ht (último estado oculto) pertencente a R128. y^​=w'hT​+b. output_shape=(qtd. registros, 1).
        neuron=128
        model = Sequential()
        model.add(LSTM(units = neuron,input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mse',optimizer=optimizer)

        model.fit(trainX, y, epochs = epochs, batch_size = batch_size,verbose=0)

         # make predictions
        y_pred_train = model.predict(trainX)
        y_pred_test = model.predict(testX)
        
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

        K.clear_session()
        import gc
        gc.collect()


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

    import numpy

    trainX = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    numpy.random.seed(1234)
    import tensorflow as tf

    y1=pd.DataFrame(y1)
    y=pd.DataFrame(y)

    y_test= sc_y.inverse_transform (y1)
    y_train= sc_y.inverse_transform (y)


    a= pd.DataFrame(a)    
    y_test= pd.DataFrame(y_test)    


    #summarize the fit of the model
    mape=numpy.mean((numpy.abs(y_test-a))/cap)*100
    classic_mape = np.mean(np.abs((y_test - a) / y_test)) * 100
    rmse= sqrt(mean_squared_error(y_test,a))
    mae=metrics.mean_absolute_error(y_test,a)

    
    print('MAPE',mape)
    print('Classic MAPE', classic_mape)
    print('RMSE',rmse)
    print('MAE',mae)

    # Save the real and predict values
    np.savetxt(os.path.join(BASE_DIR, 'y_test.txt'), y_test.values.flatten(), fmt='%.6f')
    np.savetxt(os.path.join(BASE_DIR, 'y_pred_stable_layer.txt'), a.values.flatten(), fmt='%.6f')

    dates = data1['Month'].iloc[-len(y_test):]
    dates = pd.to_datetime(dates)

    np.savetxt(os.path.join(BASE_DIR, 'dates.txt'), dates.astype(str), fmt='%s')

    return y_test.values.flatten(), a.values.flatten(), dates.astype(str).values


##Proposed Method Hybrid CEEMDAN-EWT LSTM with dropout Layer

def proposed_method_dropout_layer(new_data,i,look_back,data_partition,cap):

    x=i
    data1=new_data.loc[new_data['Month'].isin(x)]
    data1=data1.reset_index(drop=True)
    data1=data1.dropna()
    
    datas=data1['P_avg']
    datas_wind=pd.DataFrame(datas)
    dfs=datas
    s = dfs.values

    from PyEMD import EMD,EEMD,CEEMDAN
    import numpy

    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)

    IMFs = emd(s)

    full_imf=pd.DataFrame(IMFs)
    ceemdan1=full_imf.T
    
    imf1=ceemdan1.iloc[:,0]
    imf_dataps=numpy.array(imf1)
    imf_datasetss= imf_dataps.reshape(-1,1)
    imf_new_datasets=pd.DataFrame(imf_datasetss)

    import ewtpy

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

    batch_size=64
    lr=0.001
    optimizer='Adam'

    for col in new_ceemdan:

        datasetss2=pd.DataFrame(new_ceemdan[col])
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

        import numpy

        trainX = numpy.reshape(X, (X.shape[0], X.shape[1],1))
        testX = numpy.reshape(X1, (X1.shape[0], X1.shape[1],1))

        numpy.random.seed(1234)
        import tensorflow as tf
        tf.random.set_seed(1234)

        import os 
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, Activation
        from tensorflow.keras.layers import LSTM


        # LSTM Archithecture Summary:
        # - input_shape= (qtd. registros/janela temporal, características (1 neste caso)).
        # - Camada Sequencial = 128 neurônios. ([xᵢ,t-5] → [xᵢ,t-4] → [xᵢ,t-3] → [xᵢ,t-2] → [xᵢ,t-1] → [xᵢ,t] - 128 Unidades)
        # - Camada Densa = 1 neurônio. Previsao. Input: ht (último estado oculto) pertencente a R128. y^​=w'hT​+b. output_shape=(qtd. registros, 1).
        neuron=128
        model = Sequential()
        model.add(LSTM(units = neuron,input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mse',optimizer=optimizer)

        model.fit(trainX, y, epochs = epochs, batch_size = batch_size,verbose=0)

         # make predictions
        y_pred_train = model.predict(trainX)
        y_pred_test = model.predict(testX)
        
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

        K.clear_session()
        import gc
        gc.collect()


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

    import numpy

    trainX = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    numpy.random.seed(1234)
    import tensorflow as tf

    y1=pd.DataFrame(y1)
    y=pd.DataFrame(y)

    y_test= sc_y.inverse_transform (y1)
    y_train= sc_y.inverse_transform (y)


    a= pd.DataFrame(a)    
    y_test= pd.DataFrame(y_test)    


    #summarize the fit of the model
    mape=numpy.mean((numpy.abs(y_test-a))/cap)*100
    classic_mape = np.mean(np.abs((y_test - a) / y_test)) * 100
    rmse= sqrt(mean_squared_error(y_test,a))
    mae=metrics.mean_absolute_error(y_test,a)

    
    print('MAPE',mape)
    print('Classic MAPE', classic_mape)
    print('RMSE',rmse)
    print('MAE',mae)

    # Save the real and predict values
    np.savetxt(os.path.join(BASE_DIR, 'y_test.txt'), y_test.values.flatten(), fmt='%.6f')
    np.savetxt(os.path.join(BASE_DIR, 'y_pred_dropout_layer.txt'), a.values.flatten(), fmt='%.6f')

    dates = data1['Month'].iloc[-len(y_test):]
    dates = pd.to_datetime(dates)

    np.savetxt(os.path.join(BASE_DIR, 'dates.txt'), dates.astype(str), fmt='%s')

    return y_test.values.flatten(), a.values.flatten(), dates.astype(str).values


##Proposed Method Hybrid CEEMDAN-EWT LSTM with stable and dropout layer

def proposed_method_stable_and_dropout_layer(new_data,i,look_back,data_partition,cap):

    x=i
    data1=new_data.loc[new_data['Month'].isin(x)]
    data1=data1.reset_index(drop=True)
    data1=data1.dropna()
    
    datas=data1['P_avg']
    datas_wind=pd.DataFrame(datas)
    dfs=datas
    s = dfs.values

    from PyEMD import EMD,EEMD,CEEMDAN
    import numpy

    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)

    IMFs = emd(s)

    full_imf=pd.DataFrame(IMFs)
    ceemdan1=full_imf.T
    
    imf1=ceemdan1.iloc[:,0]
    imf_dataps=numpy.array(imf1)
    imf_datasetss= imf_dataps.reshape(-1,1)
    imf_new_datasets=pd.DataFrame(imf_datasetss)

    import ewtpy

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

    batch_size=64
    lr=0.001
    optimizer='Adam'

    for col in new_ceemdan:

        datasetss2=pd.DataFrame(new_ceemdan[col])
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

        import numpy

        trainX = numpy.reshape(X, (X.shape[0], X.shape[1],1))
        testX = numpy.reshape(X1, (X1.shape[0], X1.shape[1],1))

        numpy.random.seed(1234)
        import tensorflow as tf
        tf.random.set_seed(1234)

        import os 
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, Activation
        from tensorflow.keras.layers import LSTM


        # LSTM Archithecture Summary:
        # - input_shape= (qtd. registros/janela temporal, características (1 neste caso)).
        # - Camada Sequencial = 128 neurônios. ([xᵢ,t-5] → [xᵢ,t-4] → [xᵢ,t-3] → [xᵢ,t-2] → [xᵢ,t-1] → [xᵢ,t] - 128 Unidades)
        # - Camada Densa = 1 neurônio. Previsao. Input: ht (último estado oculto) pertencente a R128. y^​=w'hT​+b. output_shape=(qtd. registros, 1).
        neuron=128
        model = Sequential()
        model.add(LSTM(units = neuron,input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mse',optimizer=optimizer)

        model.fit(trainX, y, epochs = epochs, batch_size = batch_size,verbose=0)

         # make predictions
        y_pred_train = model.predict(trainX)
        y_pred_test = model.predict(testX)
        
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

    import numpy

    trainX = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    numpy.random.seed(1234)
    import tensorflow as tf

    y1=pd.DataFrame(y1)
    y=pd.DataFrame(y)

    y_test= sc_y.inverse_transform (y1)
    y_train= sc_y.inverse_transform (y)

    dates = data1['Month'].iloc[-len(y_test):]
    dates = pd.to_datetime(dates)

    a= pd.DataFrame(a)    
    y_test= pd.DataFrame(y_test)    


    #summarize the fit of the model
    mape=numpy.mean((numpy.abs(y_test-a))/cap)*100
    classic_mape = np.mean(np.abs((y_test - a) / y_test)) * 100
    rmse= sqrt(mean_squared_error(y_test,a))
    mae=metrics.mean_absolute_error(y_test,a)

    
    print('MAPE',mape)
    print('Classic MAPE', classic_mape)
    print('RMSE',rmse)
    print('MAE',mae)

    # Save the real and predict values
    np.savetxt(os.path.join(BASE_DIR, 'y_test.txt'), y_test.values.flatten(), fmt='%.6f')
    np.savetxt(os.path.join(BASE_DIR, 'y_pred_proposed_method_stable_and_dropout_layer.txt'), a.values.flatten(), fmt='%.6f')

    dates = data1['Month'].iloc[-len(y_test):]
    dates = pd.to_datetime(dates)

    np.savetxt(os.path.join(BASE_DIR, 'dates.txt'), dates.astype(str), fmt='%s')

    return y_test.values.flatten(), a.values.flatten(), dates.astype(str).values



## CEEMDAN-EWT BiLSTM
def proposed_method_with_bilstm(new_data,i,look_back,data_partition,cap):

    x=i
    data1=new_data.loc[new_data['Month'].isin(x)]
    data1=data1.reset_index(drop=True)
    data1=data1.dropna()
    
    datas=data1['P_avg']
    datas_wind=pd.DataFrame(datas)
    dfs=datas
    s = dfs.values

    from PyEMD import EMD,EEMD,CEEMDAN
    import numpy

    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)

    IMFs = emd(s)

    full_imf=pd.DataFrame(IMFs)
    ceemdan1=full_imf.T
    
    imf1=ceemdan1.iloc[:,0]
    imf_dataps=numpy.array(imf1)
    imf_datasetss= imf_dataps.reshape(-1,1)
    imf_new_datasets=pd.DataFrame(imf_datasetss)

    import ewtpy

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

    batch_size=64
    lr=0.001
    optimizer='Adam'

    for col in new_ceemdan:

        datasetss2=pd.DataFrame(new_ceemdan[col])
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

        import numpy

        trainX = numpy.reshape(X, (X.shape[0], X.shape[1],1))
        testX = numpy.reshape(X1, (X1.shape[0], X1.shape[1],1))

        numpy.random.seed(1234)
        import tensorflow as tf
        tf.random.set_seed(1234)

        
    
        import os 
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, Bidirectional
        from tensorflow.keras.layers import LSTM


        # LSTM Archithecture Summary:
        # - input_shape= (qtd. registros/janela temporal, características (1 neste caso)).
        # - Camada Sequencial = 128 neurônios. ([xᵢ,t-5] → [xᵢ,t-4] → [xᵢ,t-3] → [xᵢ,t-2] → [xᵢ,t-1] → [xᵢ,t] - 128 Unidades)
        # - Camada Densa = 1 neurônio. Previsao. Input: ht (último estado oculto) pertencente a R128. y^​=w'hT​+b. output_shape=(qtd. registros, 1).
        
        neuron = 128
        model = Sequential()
        model.add(Bidirectional(
            LSTM(units=neuron),
            input_shape=(trainX.shape[1], trainX.shape[2])
        ))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mse', optimizer=optimizer)

        model.fit(trainX, y, epochs = epochs, batch_size = batch_size,verbose=0)

         # make predictions
        y_pred_train = model.predict(trainX)
        y_pred_test = model.predict(testX)
        
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

    import numpy

    trainX = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    numpy.random.seed(1234)
    import tensorflow as tf

    y1=pd.DataFrame(y1)
    y=pd.DataFrame(y)

    y_test= sc_y.inverse_transform (y1)
    y_train= sc_y.inverse_transform (y)


    a= pd.DataFrame(a)    
    y_test= pd.DataFrame(y_test)    


    #summarize the fit of the model
    mape=numpy.mean((numpy.abs(y_test-a))/cap)*100
    classic_mape = np.mean(np.abs((y_test - a) / y_test)) * 100
    rmse= sqrt(mean_squared_error(y_test,a))
    mae=metrics.mean_absolute_error(y_test,a)

    
    print('MAPE',mape)
    print('Classic MAPE', classic_mape)
    print('RMSE',rmse)
    print('MAE',mae)

    # Save the real and predict values
    np.savetxt(os.path.join(BASE_DIR, 'y_test.txt'), y_test.values.flatten(), fmt='%.6f')
    np.savetxt(os.path.join(BASE_DIR, 'y_pred_BiLSTM.txt'), a.values.flatten(), fmt='%.6f')

    dates = data1['Month'].iloc[-len(y_test):]
    dates = pd.to_datetime(dates)

    np.savetxt(os.path.join(BASE_DIR, 'dates.txt'), dates.astype(str), fmt='%s')

    return y_test.values.flatten(), a.values.flatten(), dates.astype(str).values


# CEEMDAN-EWT GRU

def proposed_method_with_gru(new_data,i,look_back,data_partition,cap):

    x=i
    data1=new_data.loc[new_data['Month'].isin(x)]
    data1=data1.reset_index(drop=True)
    data1=data1.dropna()

    dates = data1.Date
    
    datas=data1['P_avg']
    datas_wind=pd.DataFrame(datas)
    dfs=datas
    s = dfs.values

    from PyEMD import EMD,EEMD,CEEMDAN
    import numpy

    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)

    IMFs = emd(s)

    full_imf=pd.DataFrame(IMFs)
    ceemdan1=full_imf.T
    
    imf1=ceemdan1.iloc[:,0]
    imf_dataps=numpy.array(imf1)
    imf_datasetss= imf_dataps.reshape(-1,1)
    imf_new_datasets=pd.DataFrame(imf_datasetss)

    import ewtpy

    ewt,  mfb ,boundaries = ewtpy.EWT1D(imf1, N =3)
    df_ewt=pd.DataFrame(ewt)

    df_ewt.drop(df_ewt.columns[2],axis=1,inplace=True)
    denoised=df_ewt.sum(axis = 1, skipna = True) 
    ceemdan_without_imf1=ceemdan1.iloc[:,1:]
    new_ceemdan=pd.concat([denoised,ceemdan_without_imf1],axis=1)
    new_ceemdan.index = dates
    
    pred_test=[]
    test_ori=[]
    pred_train=[]
    train_ori=[]

    batch_size=64
    lr=0.001
    optimizer='Adam'

    for col in new_ceemdan:

        datasetss2=pd.DataFrame(new_ceemdan[col])
        datasets=datasetss2.values
        dates = new_ceemdan.index.values

        train_size = int(len(datasets) * data_partition)
        test_size = len(datasets) - train_size
        train, test = datasets[0:train_size], datasets[train_size:len(datasets)]
        train_dates, test_dates = dates[:train_size], dates[train_size:]

        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)

        train_dates_aligned = train_dates[look_back+1:]
        test_dates_aligned = test_dates[look_back+1:]

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

        import numpy

        trainX = numpy.reshape(X, (X.shape[0], X.shape[1],1))
        testX = numpy.reshape(X1, (X1.shape[0], X1.shape[1],1))

        numpy.random.seed(1234)
        import tensorflow as tf
        tf.random.set_seed(1234)

        
    
        import os 
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, Activation
        from tensorflow.keras.layers import GRU


        # LSTM Archithecture Summary:
        # - input_shape= (qtd. registros/janela temporal, características (1 neste caso)).
        # - Camada Sequencial = 128 neurônios. ([xᵢ,t-5] → [xᵢ,t-4] → [xᵢ,t-3] → [xᵢ,t-2] → [xᵢ,t-1] → [xᵢ,t] - 128 Unidades)
        # - Camada Densa = 1 neurônio. Previsao. Input: ht (último estado oculto) pertencente a R128. y^​=w'hT​+b. output_shape=(qtd. registros, 1).
        neuron=128
        model = Sequential()
        model.add(GRU(units = neuron,input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mse',optimizer=optimizer)

        model.fit(trainX, y, epochs = epochs, batch_size = batch_size,verbose=0)

         # make predictions
        y_pred_train = model.predict(trainX)
        y_pred_test = model.predict(testX)
        
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

    import numpy

    trainX = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    numpy.random.seed(1234)
    import tensorflow as tf

    y1=pd.DataFrame(y1)
    y=pd.DataFrame(y)

    y_test= sc_y.inverse_transform (y1)
    y_train= sc_y.inverse_transform (y)


    a= pd.DataFrame(a)    
    y_test= pd.DataFrame(y_test)    


    #summarize the fit of the model
    mape=numpy.mean((numpy.abs(y_test-a))/cap)*100
    classic_mape = np.mean(np.abs((y_test - a) / y_test)) * 100
    rmse= sqrt(mean_squared_error(y_test,a))
    mae=metrics.mean_absolute_error(y_test,a)

    
    print('MAPE',mape)
    print('Classic MAPE', classic_mape)
    print('RMSE',rmse)
    print('MAE',mae)

    # Save the real and predict values
    np.savetxt(os.path.join(BASE_DIR, 'y_test.txt'), y_test.values.flatten(), fmt='%.6f')
    np.savetxt(os.path.join(BASE_DIR, 'y_pred_gru.txt'), a.values.flatten(), fmt='%.6f')

    dates = pd.to_datetime(test_dates_aligned)

    np.savetxt(os.path.join(BASE_DIR, 'dates.txt'), dates.astype(str), fmt='%s')

    return y_test.values.flatten(), a.values.flatten(), dates.astype(str).values

## CEEMDAN-EWT BiGRU

def proposed_method_with_bigru(new_data,i,look_back,data_partition,cap):

    x=i
    data1=new_data.loc[new_data['Month'].isin(x)]
    data1=data1.reset_index(drop=True)
    data1=data1.dropna()
    
    datas=data1['P_avg']
    datas_wind=pd.DataFrame(datas)
    dfs=datas
    s = dfs.values

    from PyEMD import EMD,EEMD,CEEMDAN
    import numpy

    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)

    IMFs = emd(s)

    full_imf=pd.DataFrame(IMFs)
    ceemdan1=full_imf.T
    
    imf1=ceemdan1.iloc[:,0]
    imf_dataps=numpy.array(imf1)
    imf_datasetss= imf_dataps.reshape(-1,1)
    imf_new_datasets=pd.DataFrame(imf_datasetss)

    import ewtpy

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

    batch_size=64
    lr=0.001
    optimizer='Adam'

    for col in new_ceemdan:

        datasetss2=pd.DataFrame(new_ceemdan[col])
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

        import numpy

        trainX = numpy.reshape(X, (X.shape[0], X.shape[1],1))
        testX = numpy.reshape(X1, (X1.shape[0], X1.shape[1],1))

        numpy.random.seed(1234)
        import tensorflow as tf
        tf.random.set_seed(1234)

        
    
        import os 
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, Bidirectional
        from tensorflow.keras.layers import GRU


        # LSTM Archithecture Summary:
        # - input_shape= (qtd. registros/janela temporal, características (1 neste caso)).
        # - Camada Sequencial = 128 neurônios. ([xᵢ,t-5] → [xᵢ,t-4] → [xᵢ,t-3] → [xᵢ,t-2] → [xᵢ,t-1] → [xᵢ,t] - 128 Unidades)
        # - Camada Densa = 1 neurônio. Previsao. Input: ht (último estado oculto) pertencente a R128. y^​=w'hT​+b. output_shape=(qtd. registros, 1).
        
        neuron = 128
        model = Sequential()
        model.add(Bidirectional(
            GRU(units=neuron),
            input_shape=(trainX.shape[1], trainX.shape[2])
        ))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mse', optimizer=optimizer)

        model.fit(trainX, y, epochs = epochs, batch_size = batch_size,verbose=0)

         # make predictions
        y_pred_train = model.predict(trainX)
        y_pred_test = model.predict(testX)
        
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

    import numpy

    trainX = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    numpy.random.seed(1234)
    import tensorflow as tf

    y1=pd.DataFrame(y1)
    y=pd.DataFrame(y)

    y_test= sc_y.inverse_transform (y1)
    y_train= sc_y.inverse_transform (y)


    a= pd.DataFrame(a)    
    y_test= pd.DataFrame(y_test)    


    #summarize the fit of the model
    mape=numpy.mean((numpy.abs(y_test-a))/cap)*100
    classic_mape = np.mean(np.abs((y_test - a) / y_test)) * 100
    rmse= sqrt(mean_squared_error(y_test,a))
    mae=metrics.mean_absolute_error(y_test,a)

    
    print('MAPE',mape)
    print('Classic MAPE', classic_mape)
    print('RMSE',rmse)
    print('MAE',mae)

    # Save the real and predict values
    np.savetxt(os.path.join(BASE_DIR, 'y_test.txt'), y_test.values.flatten(), fmt='%.6f')
    np.savetxt(os.path.join(BASE_DIR, 'y_pred_BiGRU.txt'), a.values.flatten(), fmt='%.6f')

    dates = data1['Month'].iloc[-len(y_test):]
    dates = pd.to_datetime(dates)

    np.savetxt(os.path.join(BASE_DIR, 'dates.txt'), dates.astype(str), fmt='%s')

    return y_test.values.flatten(), a.values.flatten(), dates.astype(str).values


# CEEMDAN-EWT Transformer with Keras

def proposed_method_with_transformer_keras(new_data, i, look_back, data_partition, cap):

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
            import tensorflow as tf

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

    # ===============================
    # 1. Seleção dos dados
    # ===============================
    data1 = new_data.loc[new_data['Month'].isin(i)]
    data1 = data1.reset_index(drop=True).dropna()

    dfs = data1['P_avg']
    s = dfs.values

    # ===============================
    # 2. CEEMDAN
    # ===============================
    from PyEMD import CEEMDAN
    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)

    IMFs = emd(s)
    ceemdan1 = pd.DataFrame(IMFs).T

    # ===============================
    # 3. EWT
    # ===============================
    import ewtpy

    imf1 = ceemdan1.iloc[:, 0]
    ewt, _, _ = ewtpy.EWT1D(imf1, N=3)

    df_ewt = pd.DataFrame(ewt)
    df_ewt.drop(df_ewt.columns[2], axis=1, inplace=True)

    denoised = df_ewt.sum(axis=1)
    ceemdan_without_imf1 = ceemdan1.iloc[:, 1:]

    new_ceemdan = pd.concat([denoised, ceemdan_without_imf1], axis=1)

    # ===============================
    # 4. Loop por componente
    # ===============================
    pred_test = []

    for col in new_ceemdan:

        dataset = new_ceemdan[col].values.reshape(-1, 1)

        train_size = int(len(dataset) * data_partition)
        train, test = dataset[:train_size], dataset[train_size:]

        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)

        sc_X = StandardScaler()
        sc_y = StandardScaler()

        X_train = sc_X.fit_transform(trainX)
        y_train = sc_y.fit_transform(trainY.reshape(-1, 1)).ravel()

        X_test = sc_X.transform(testX)
        y_test = sc_y.transform(testY.reshape(-1, 1)).ravel()

        trainX = X_train.reshape(X_train.shape[0], look_back, 1)
        testX = X_test.reshape(X_test.shape[0], look_back, 1)

        # ===============================
        # 5. Modelo Transformer
        # ===============================
        inputs = tf.keras.Input(shape=(look_back, 1))

        x = tf.keras.layers.Dense(32)(inputs)
        x = PositionalEncoding(32)(x)

        attn = MultiHeadAttention(embed_dim=32, num_heads=4)(x)
        attn = tf.keras.layers.Dropout(0.1)(attn)

        x = tf.keras.layers.Add()([x, attn])
        x = tf.keras.layers.LayerNormalization()(x)

        ff = tf.keras.layers.Dense(64, activation="relu")(x)
        ff = tf.keras.layers.Dense(32)(ff)
        ff = tf.keras.layers.Dropout(0.2)(ff)

        x = tf.keras.layers.Add()([x, ff])
        x = tf.keras.layers.LayerNormalization()(x)

        x = tf.keras.layers.Lambda(
            lambda t: t[:, -1, :]
        )(x)

        outputs = tf.keras.layers.Dense(1)(x)

        model = tf.keras.Model(inputs, outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss="mse"
        )

        model.fit(
            trainX, y_train,
            epochs=epochs,
            batch_size=64,
            verbose=0
        )

        # ===============================
        # 6. Previsões
        # ===============================
        y_pred_test = model.predict(testX).ravel()
        y_pred_test = sc_y.inverse_transform(y_pred_test.reshape(-1, 1))

        pred_test.append(pd.DataFrame(y_pred_test))

    # ===============================
    # 7. Reconstrução
    # ===============================
    pred_test_matrix = np.hstack([df.values for df in pred_test])
    a = pred_test_matrix.sum(axis=1).reshape(-1, 1)

    # ===============================
    # 8. Métricas
    # ===============================
    dataset = dfs.values.reshape(-1, 1)
    train_size = int(len(dataset) * data_partition)
    test = dataset[train_size:]

    _, testY = create_dataset(test, look_back)
    y_test = testY.reshape(-1, 1)

    mape = np.mean(np.abs(y_test - a) / cap) * 100
    rmse = sqrt(mean_squared_error(y_test, a))
    mae = metrics.mean_absolute_error(y_test, a)

    print("MAPE:", mape)
    print("RMSE:", rmse)
    print("MAE:", mae)

    # Save the real and predict values
    np.savetxt(os.path.join(BASE_DIR, 'y_test.txt'), y_test.flatten(), fmt='%.6f')
    np.savetxt(os.path.join(BASE_DIR, 'y_pred_transformer.txt'), a.flatten(), fmt='%.6f')

    dates = data1['Month'].iloc[-len(y_test):]
    dates = pd.to_datetime(dates)

    np.savetxt(os.path.join(BASE_DIR, 'dates.txt'), dates.astype(str), fmt='%s')

    return y_test.flatten(), a.flatten(), dates.astype(str).values


# CEEMDAN-EWT PatchTransformer with TensorFlow

def proposed_method_with_patchtransformer_tf(new_data,i,look_back,data_partition,cap):

    x=i
    data1=new_data.loc[new_data['Month'].isin(x)]
    data1=data1.reset_index(drop=True)
    data1=data1.dropna()
    
    datas=data1['P_avg']
    datas_wind=pd.DataFrame(datas)
    dfs=datas
    s = dfs.values

    from PyEMD import EMD,EEMD,CEEMDAN
    import numpy

    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)

    IMFs = emd(s)

    full_imf=pd.DataFrame(IMFs)
    ceemdan1=full_imf.T
    
    imf1=ceemdan1.iloc[:,0]
    imf_dataps=numpy.array(imf1)
    imf_datasetss= imf_dataps.reshape(-1,1)
    imf_new_datasets=pd.DataFrame(imf_datasetss)

    import ewtpy

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

    batch_size=64
    lr=0.001
    optimizer='Adam'

    for col in new_ceemdan:

        datasetss2=pd.DataFrame(new_ceemdan[col])
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

        import numpy

        trainX = numpy.reshape(X, (X.shape[0], X.shape[1],1))
        testX = numpy.reshape(X1, (X1.shape[0], X1.shape[1],1))

        numpy.random.seed(1234)
        import tensorflow as tf
        tf.random.set_seed(1234)

        import os 
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader

        # Positional Encoding clássico (sinusoidal)
        class PositionalEncoding(nn.Module):
            def __init__(self, d_model, max_len=500):
                super().__init__()
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0)
                self.register_buffer("pe", pe)

            def forward(self, x):
                x = x + self.pe[:, :x.size(1)]
                return x

        class PatchEmbedding(nn.Module):
            def __init__(self, input_dim, patch_len, embed_dim):
                super().__init__()
                self.patch_len = patch_len
                self.proj = nn.Linear(input_dim * patch_len, embed_dim)
                self.norm = nn.LayerNorm(embed_dim)

            def forward(self, x):
                B, T, D = x.shape
                # Gera patches
                x = x.view(B, T // self.patch_len, self.patch_len * D)
                x = self.proj(x)
                x = self.norm(x)
                return x  # [B, num_patches, embed_dim]

        class PatchTST(nn.Module):
            def __init__(self, input_dim=1, patch_len=3, embed_dim=16, num_heads=2,
                        num_layers=1, pred_len=1, dropout=0.1):
                super().__init__()
                self.embedding = PatchEmbedding(input_dim, patch_len, embed_dim)
                # Positional Encoding
                self.pos_encoder = PositionalEncoding(embed_dim)
                # Transformer Encoder 
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim, nhead=num_heads,
                    dropout=dropout, batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                # Head
                self.head = nn.Linear(embed_dim, pred_len)

            def forward(self, x):
                # x: [B, T, D]
                x = self.embedding(x)        # [B, num_patches, embed_dim]
                x = self.pos_encoder(x)
                x = self.transformer(x)
                # Pooling pelo último patch (melhor para janela curta)
                x = x[:, -1, :]              # [B, embed_dim]
                return self.head(x)          # [B, pred_len]


        class TimeSeriesDataset(torch.utils.data.Dataset):
            def __init__(self, X, y):
                self.X = torch.tensor(X, dtype=torch.float32)
                self.y = torch.tensor(y, dtype=torch.float32)

            def __len__(self):
                return len(self.X)

            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]

        # Patch (Time-series) Transformer Archithecture:

        train_dataset = TimeSeriesDataset(trainX, y)
        test_dataset  = TimeSeriesDataset(testX, y1)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = PatchTST(
            input_dim=1,
            seq_len=trainX.shape[1],
            patch_len=3,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            pred_len=1
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)

                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

        model.eval()

        import numpy as np

        # ---- TRAIN PREDICTIONS ----
        preds_train = []
        with torch.no_grad():
            for xb, _ in train_loader:
                xb = xb.to(device)
                out = model(xb)
                preds_train.append(out.cpu().numpy())

        y_pred_train = np.concatenate(preds_train).ravel()

        # ---- TEST PREDICTIONS ----
        preds_test = []
        with torch.no_grad():
            for xb, _ in test_loader:
                xb = xb.to(device)
                out = model(xb)
                preds_test.append(out.cpu().numpy())

        y_pred_test = np.concatenate(preds_test).ravel()


        y_pred_test  = pd.DataFrame(y_pred_test)
        y_pred_train = pd.DataFrame(y_pred_train)

        y1 = pd.DataFrame(y1)
        y  = pd.DataFrame(y)

        y_test  = sc_y.inverse_transform(y1)
        y_train = sc_y.inverse_transform(y)

        y_pred_test1  = sc_y.inverse_transform(y_pred_test)
        y_pred_train1 = sc_y.inverse_transform(y_pred_train)


        pred_test.append(y_pred_test1)
        test_ori.append(y_test)

        pred_train.append(y_pred_train1)
        train_ori.append(y_train)

    
    result_pred_test  = pd.DataFrame.from_records(pred_test)
    result_pred_train = pd.DataFrame.from_records(pred_train)

    a = result_pred_test.sum(axis=0, skipna=True)
    b = result_pred_train.sum(axis=0, skipna=True)

    a = pd.DataFrame(a)
    y_test = pd.DataFrame(y_test)

    mape = np.mean((np.abs(y_test - a)) / cap) * 100
    rmse = sqrt(mean_squared_error(y_test, a))
    mae  = metrics.mean_absolute_error(y_test, a)

    print('MAPE', mape)
    print('RMSE', rmse)
    print('MAE', mae)

    # Save the real and predict values
    np.savetxt(os.path.join(BASE_DIR, 'y_test.txt'), y_test.values.flatten(), fmt='%.6f')
    np.savetxt(os.path.join(BASE_DIR, 'y_pred_patch_transformer.txt'), a.values.flatten(), fmt='%.6f')

    dates = data1['Month'].iloc[-len(y_test):]
    dates = pd.to_datetime(dates)

    np.savetxt(os.path.join(BASE_DIR, 'dates.txt'), dates.astype(str), fmt='%s')

    return y_test.values.flatten(), a.values.flatten(), dates.astype(str).values

# Hybrid CEEMDAN-EWT KAN

def proposed_method_with_kan(
    new_data, i, look_back, data_partition, cap,
    epochs=epochs, batch_size=32, Q=4
):
    import numpy as np
    import pandas as pd
    from math import sqrt
    from sklearn.metrics import mean_squared_error
    from sklearn import metrics
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


    # -----------------------------
    # Seleção dos dados
    # -----------------------------
    x = i
    data1 = new_data.loc[new_data['Month'].isin(x)]
    data1 = data1.reset_index(drop=True).dropna()

    dfs = data1['P_avg']
    s = dfs.values

    s = np.asarray(dfs, dtype=np.float64).flatten()

    # -----------------------------
    # CEEMDAN
    # -----------------------------
    from PyEMD import CEEMDAN
    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)
    IMFs = emd(s)

    ceemdan1 = pd.DataFrame(IMFs).T

    # -----------------------------
    # EWT no IMF1
    # -----------------------------
    import ewtpy
    imf1 = ceemdan1.iloc[:, 0]
    ewt, _, _ = ewtpy.EWT1D(imf1, N=3)

    df_ewt = pd.DataFrame(ewt)
    df_ewt.drop(df_ewt.columns[2], axis=1, inplace=True)
    denoised = df_ewt.sum(axis=1)

    ceemdan_without_imf1 = ceemdan1.iloc[:, 1:]
    new_ceemdan = pd.concat([denoised, ceemdan_without_imf1], axis=1)

    # -----------------------------
    # Containers
    # -----------------------------
    pred_test = []

    # -----------------------------
    # Loop por IMF
    # -----------------------------
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
    # Ground truth
    # -----------------------------
    dataset = dfs.values.reshape(-1, 1)
    _, testY = create_dataset(dataset, look_back)

    split = int(len(testY) * data_partition)
    y_test = testY[split:].reshape(-1, 1)

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

    dates = data1['Month'].iloc[-len(y_test):]
    dates = pd.to_datetime(dates)

    np.savetxt(os.path.join(BASE_DIR, 'dates.txt'), dates.astype(str), fmt='%s')

    return  y_test.values.flatten(), a.values.flatten(), dates.astype(str).values


def proposed_method_with_deeponet(
    new_data, i, look_back, data_partition, cap,
    epochs=50, batch_size=32, hidden_dim=64
):
    import numpy as np
    import pandas as pd
    from PyEMD import CEEMDAN
    import ewtpy
    from sklearn.metrics import mean_squared_error
    from sklearn import metrics
    from math import sqrt
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

    # -----------------------------
    # Seleção dos dados
    # -----------------------------
    data1 = new_data.loc[new_data['Month'].isin(i)].dropna().reset_index(drop=True)
    dfs = data1['P_avg']
    s = dfs.values

    # -----------------------------
    # CEEMDAN
    # -----------------------------
    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)
    IMFs = emd(s)
    ceemdan = pd.DataFrame(IMFs).T

    # -----------------------------
    # EWT no IMF1
    # -----------------------------
    imf1 = ceemdan.iloc[:, 0]
    ewt, _, _ = ewtpy.EWT1D(imf1, N=3)
    denoised = pd.DataFrame(ewt).iloc[:, :2].sum(axis=1)

    new_ceemdan = pd.concat([denoised, ceemdan.iloc[:, 1:]], axis=1)

    pred_test = []

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

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
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

    # Ground truth
    dataset = dfs.values.reshape(-1, 1)
    _, testY = create_dataset(dataset, look_back)
    y_test = testY[int(len(testY)*data_partition):].reshape(-1, 1)

    dates = data1['Month'].iloc[-len(y_test):]
    dates = pd.to_datetime(dates)

    # -----------------------------
    # Métricas
    # -----------------------------
    mape = np.mean(np.abs(y_test - a) / cap) * 100
    rmse = sqrt(mean_squared_error(y_test, a))
    mae  = metrics.mean_absolute_error(y_test, a)

    print("MAPE:", mape)
    print("RMSE:", rmse)
    print("MAE:", mae)

    return y_test.values.flatten(), a.values.flatten(), dates.astype(str).values


def proposed_method_with_lstm_deeponet(
    new_data, i, look_back, data_partition, cap,
    epochs=50, batch_size=32, hidden_dim=64
):

    import numpy as np
    import pandas as pd
    from PyEMD import CEEMDAN
    import ewtpy
    from sklearn.metrics import mean_squared_error
    from sklearn import metrics
    from math import sqrt
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

    # ==========================================================
    # Seleção dos dados
    # ==========================================================
    data1 = new_data.loc[new_data['Month'].isin(i)].dropna().reset_index(drop=True)
    dfs = data1['P_avg']
    s = dfs.values

    # ==========================================================
    # CEEMDAN
    # ==========================================================
    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)
    IMFs = emd(s)
    ceemdan = pd.DataFrame(IMFs).T

    # ==========================================================
    # EWT no IMF1
    # ==========================================================
    imf1 = ceemdan.iloc[:, 0]
    ewt, _, _ = ewtpy.EWT1D(imf1, N=3)
    denoised = pd.DataFrame(ewt).iloc[:, :2].sum(axis=1)

    new_ceemdan = pd.concat([denoised, ceemdan.iloc[:, 1:]], axis=1)

    pred_test = []

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

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
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
    
    dates = data1['Month'].iloc[-len(y_test):]
    dates = pd.to_datetime(dates)

    # Ground truth
    dataset = dfs.values.reshape(-1, 1)
    _, testY = create_dataset(dataset, look_back)
    y_test = testY[int(len(testY) * data_partition):].reshape(-1, 1)

    # ==========================================================
    # Métricas
    # ==========================================================
    mape = np.mean(np.abs(y_test - a) / cap) * 100
    rmse = sqrt(mean_squared_error(y_test, a))
    mae  = metrics.mean_absolute_error(y_test, a)

    print("MAPE:", mape)
    print("RMSE:", rmse)
    print("MAE:", mae)

    return y_test.values.flatten(), a.values.flatten(), dates.astype(str).values


if __name__=='__main__':
    import pandas as pd
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