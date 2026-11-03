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

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

from seqKAN_public.seqKAN import SeqKANSeq

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
# HYBRID MODELS
# =============================================================================

##Proposed Method Hybrid CEEMDAN-EWT LSTM

# ============================================================
# Proposed Method Hybrid CEEMDAN-EWT-SEQKAN
# ============================================================

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

    print(
        "Componentes para previsão:",
        new_ceemdan.shape[1],
        flush=True
    )

    # ============================================================
    # 3. PREPARAÇÃO DAS ENTRADAS
    # ============================================================

    # Após a IMF 9, a transformação de Hilbert pode não produzir
    # amplitudes instantâneas válidas para determinadas componentes.
    # Nesses casos, utiliza-se a própria componente como entrada.

    if hilbert:

        print("Aplicando HHT...", flush=True)

        sample_rate = 1.0  # Verificar conforme a frequência dos dados

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

            # Verifica frequências e amplitudes instantâneas válidas.
            valid = (
                np.isfinite(component_if) &
                np.isfinite(component_ia) &
                (component_ia > 1e-12)
            )

            if not np.any(valid):

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

        print(
            "Utilizando componentes originais como entradas...",
            flush=True
        )

        input_components = [
            new_ceemdan.iloc[:, i].to_numpy(
                dtype=np.float32
            ).reshape(-1, 1)
            for i in range(new_ceemdan.shape[1])
        ]

    # ============================================================
    # 4. CONFIGURAÇÃO DO SEQKAN
    # ============================================================

    # Dispositivo
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Dispositivo utilizado: {device}", flush=True)

    # Parâmetros do KAN original utilizado pelo SeqKAN
    kan_params = {
        "cell": {
            "grid": 20,
            "k": 3,
            "grid_range": [-10, 10],
            "grid_eps": 0.02,
            "seed": 42,
            "sp_trainable": False,
            "sb_trainable": False,
            "affine_trainable": False,
            "symbolic_enabled": False,
            "sparse_init": False
        }
    }

    # ============================================================
    # 5. PREVISÃO INDIVIDUAL DE CADA COMPONENTE
    # ============================================================

    print(
        "Início do treinamento do modelo SeqKAN",
        flush=True
    )

    pred_test = []
    pred_train = []

    for idx, col in enumerate(new_ceemdan.columns):

        print(
            f"\nProcessando componente {idx + 1}",
            flush=True
        )

        # --------------------------------------------------------
        # Entrada: HHT ou componente original
        # --------------------------------------------------------

        features = np.asarray(
            input_components[idx],
            dtype=np.float32
        )

        # Alvo: sempre a componente original
        target = new_ceemdan[col].to_numpy(
            dtype=np.float32
        )

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

        trainY = np.asarray(
            trainY,
            dtype=np.float32
        ).reshape(-1, 1)

        testY = np.asarray(
            testY,
            dtype=np.float32
        ).reshape(-1, 1)

        if len(trainX) == 0 or len(testX) == 0:
            raise ValueError(
                f"Quantidade insuficiente de amostras na componente {idx + 1}."
            )

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
        trainX = np.ascontiguousarray(
            X,
            dtype=np.float32
        )

        testX = np.ascontiguousarray(
            X1,
            dtype=np.float32
        )

        y = np.ascontiguousarray(
            y.reshape(-1, 1),
            dtype=np.float32
        )

        y1 = np.ascontiguousarray(
            y1.reshape(-1, 1),
            dtype=np.float32
        )

        print(
            f"trainX: {trainX.shape} | "
            f"testX: {testX.shape} | "
            f"Features: {n_features}",
            flush=True
        )

        # --------------------------------------------------------
        # Construção do SeqKAN
        # --------------------------------------------------------

        gc.collect()

        np.random.seed(1234)
        torch.manual_seed(1234)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1234)

        model = SeqKANSeq(
            input_size=trainX.shape[2],
            hidden_size=neuron,
            output_size=1,
            kan_params=kan_params,
            device=device
        ).to(device)

        # --------------------------------------------------------
        # Conversão para tensores PyTorch
        # --------------------------------------------------------

        trainX_tensor = torch.from_numpy(trainX)
        testX_tensor = torch.from_numpy(testX)

        y_tensor = torch.from_numpy(y)

        # --------------------------------------------------------
        # DataLoader
        # --------------------------------------------------------

        train_dataset = TensorDataset(
            trainX_tensor,
            y_tensor
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        # --------------------------------------------------------
        # Função de perda e otimizador
        # --------------------------------------------------------

        criterion = nn.MSELoss()

        optimizer_torch = torch.optim.Adam(
            model.parameters(),
            lr=0.001
        )

        # --------------------------------------------------------
        # Treinamento
        # --------------------------------------------------------

        model.train()

        for epoch in range(epochs):

            epoch_loss = 0.0

            for batch_X, batch_y in train_loader:

                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                optimizer_torch.zero_grad()

                # Saída do SeqKAN: (batch, look_back, output_size)
                outputs = model(batch_X)

                # Utiliza a última saída temporal
                outputs = outputs[:, -1, :]

                loss = criterion(
                    outputs,
                    batch_y
                )

                loss.backward()

                optimizer_torch.step()

                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0:

                print(
                    f"Componente {idx + 1} | "
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Loss: {epoch_loss / len(train_loader):.6f}",
                    flush=True
                )

        # --------------------------------------------------------
        # Previsões
        # --------------------------------------------------------

        model.eval()

        with torch.no_grad():

            trainX_device = trainX_tensor.to(device)
            testX_device = testX_tensor.to(device)

            y_pred_train = model(
                trainX_device
            )[:, -1, :].cpu().numpy()

            y_pred_test = model(
                testX_device
            )[:, -1, :].cpu().numpy()

        # --------------------------------------------------------
        # Retorna à escala original
        # --------------------------------------------------------

        y_pred_train1 = sc_y.inverse_transform(
            y_pred_train.reshape(-1, 1)
        )

        y_pred_test1 = sc_y.inverse_transform(
            y_pred_test.reshape(-1, 1)
        )

        pred_train.append(y_pred_train1)
        pred_test.append(y_pred_test1)

        # --------------------------------------------------------
        # Liberação de memória
        # --------------------------------------------------------

        del model
        del optimizer_torch
        del criterion
        del train_loader
        del train_dataset
        del trainX_tensor
        del testX_tensor
        del y_tensor
        del trainX_device
        del testX_device

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

    # ============================================================
    # 6. RECONSTRUÇÃO DO SINAL
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
    # 7. VALORES REAIS DO SINAL ORIGINAL
    # ============================================================

    datasets = np.asarray(datasets).reshape(-1)

    train_size = int(len(datasets) * data_partition)

    train = datasets[:train_size]
    test = datasets[train_size:]

    # Alinhamento com as previsões
    y_test = test[look_back:look_back + len(a)]
    y_train = train[look_back:look_back + len(b)]

    # ============================================================
    # 8. MÉTRICAS
    # ============================================================

    mask = y_test != 0

    mape = np.mean(
        np.abs(y_test - a)
    ) / cap * 100

    classic_mape = np.mean(
        np.abs(
            (y_test[mask] - a[mask]) / y_test[mask]
        )
    ) * 100

    rmse = sqrt(
        mean_squared_error(y_test, a)
    )

    mae = metrics.mean_absolute_error(
        y_test,
        a
    )

    print('MAPE', mape)
    print('Classic MAPE', classic_mape)
    print('RMSE', rmse)
    print('MAE', mae)

    return y_test.flatten(), a.flatten()


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