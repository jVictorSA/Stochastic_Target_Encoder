import copy
from sklearn.model_selection import train_test_split
import pandas as pd
pd.options.display.float_format = '{:.2f}'.format

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

import math


def encode_rbte(dados_treino, dados_teste, encoder:map, coluna_alvo:str):
    treino = copy.deepcopy(dados_treino)
    teste = copy.deepcopy(dados_teste)

    encoded_treino = encoder.fit_transform(treino, treino[coluna_alvo])
    encoded_teste = encoder.transform(teste)

    return encoded_treino, encoded_teste


def split_treino_val(dataframe, coluna_alvo:str):
    # dataframe de split dos dados
    data_split = copy.deepcopy(dataframe)

    # Dataframes de atributos independentes e dependentes
    X = data_split.drop([coluna_alvo], axis=1)
    y = data_split[coluna_alvo]

    # Split de dados de treinamento e teste
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=42)


    # Concatenar as variáveis independentes com a dependente dos conjuntos de treino e teste
    treino = pd.concat([X_treino, y_treino], axis=1)

    # Concatenar as variáveis independentes com a dependente dos conjuntos de treino e teste
    teste = pd.concat([X_teste, y_teste], axis=1)

    return treino, teste

def encode_rivais(dados_treino, dados_teste, rivais:map, coluna_alvo:str):
    treino = copy.deepcopy(dados_treino)
    teste = copy.deepcopy(dados_teste)
    alvos_treino = treino[coluna_alvo]
    alvos_teste = teste[coluna_alvo]
    treino.drop([coluna_alvo], axis=1, inplace=True)
    teste.drop([coluna_alvo], axis=1, inplace=True)
    atributos = list(treino.columns)

    encoded_dataframes_treino = {}
    encoded_dataframes_teste  = {}

    for key, valor in rivais.items():
        encoded_dataframes_treino[key] = valor.fit_transform(treino[atributos], alvos_treino)
        encoded_dataframes_teste[key]  = valor.transform(teste [atributos])

        encoded_dataframes_treino[key][coluna_alvo] = alvos_treino
        encoded_dataframes_teste[key][coluna_alvo]  = alvos_teste


    return encoded_dataframes_treino, encoded_dataframes_teste

def get_max_categorical_value_count_in_data(dataframe):
    cols = dataframe.columns
    num_cols = dataframe._get_numeric_data().columns

    categoric_cols = list(set(cols) - set(num_cols))

    valores_categoricos = {}
    catecoric_values_count = {}

    for coluna in categoric_cols:
        valores = dataframe[coluna].unique()
        valores_categoricos[coluna] = valores
        catecoric_values_count[coluna] = {}
        catecoric_values_count[coluna] = dataframe[coluna].value_counts()

    max_count = dict(category = None, value = None, value_count = 0)
    for chave, valor in catecoric_values_count.items():
        if(valor.max() > max_count["value_count"]):
            max_count["category"] = chave
            max_count["value"] = valor.idxmax()
            max_count["value_count"] = valor.max()

    return max_count

def get_smoothing_parameter(dataframe):
    max_cat_value_count = get_max_categorical_value_count_in_data(dataframe)

    return max_cat_value_count, \
           max_cat_value_count["value_count"] * 1.25, \
           max_cat_value_count["value_count"] * 1.50, \
           max_cat_value_count["value_count"] * 2, \
           max_cat_value_count["value_count"] * 3

def normalizar(dados_treino, dados_teste, coluna_alvo):

    scaler = MinMaxScaler()

    treino_to_fit = copy.deepcopy(dados_treino.drop([coluna_alvo], axis=1))
    teste_to_fit  = copy.deepcopy(dados_teste.drop([coluna_alvo], axis=1))
    colunas = treino_to_fit.columns

    # Normaliza os dados
    treino_to_fit[colunas] = scaler.fit_transform(treino_to_fit[colunas])
    teste_to_fit[colunas]  = scaler.transform(teste_to_fit[colunas])

    treino_to_fit[coluna_alvo] = dados_treino[coluna_alvo]
    teste_to_fit[coluna_alvo]  = dados_teste[coluna_alvo]

    return treino_to_fit, teste_to_fit

def data_split(dataframe, coluna_alvo):
    dados_split = copy.deepcopy(dataframe)

    features = dados_split.drop([coluna_alvo], axis=1)
    alvo  = dados_split[coluna_alvo]

    return features, alvo

def medias_scores(metricas_RBTE, metricas_rivais):
    medias_ste = {}
    for key, valor in metricas_RBTE.items():
        medias_ste[key] = valor.mean(0)
        medias_ste[key] = pd.DataFrame({'RMSE':[medias_ste[key]['RMSE']],'MAE':[medias_ste[key]['MAE']]}, index=[f"STE {key}"])

    medias_rivais = {}
    for key, valor in metricas_rivais.items():
        medias_rivais[key] = valor.mean(0)
        medias_rivais[key] = pd.DataFrame({'RMSE':[medias_rivais[key]['RMSE']],'MAE':[medias_rivais[key]['MAE']]}, index=[key])

    return medias_ste, medias_rivais

def desvios_scores(metricas_RBTE, metricas_rivais):
    desvios_ste = {}
    for key, valor in metricas_RBTE.items():
        desvios_ste[key] = valor.std(0)
        desvios_ste[key] = pd.DataFrame({'RMSE':[desvios_ste[key]['RMSE']],'MAE':[desvios_ste[key]['MAE']]}, index=[f"STE {key}"])

    desvios_rivais = {}
    for key, valor in metricas_rivais.items():
        desvios_rivais[key] = valor.std(0)
        desvios_rivais[key] = pd.DataFrame({'RMSE':[desvios_rivais[key]['RMSE']],'MAE':[desvios_rivais[key]['MAE']]}, index=[key])

    return desvios_ste, desvios_rivais

def metricas_to_DataFrame(scores_RBTE, scores_rivais, model_name):
    metricas = pd.DataFrame()

    for key, valor in scores_RBTE.items():
        ste = pd.DataFrame(valor.values, columns=valor.columns, index=[f'STE_{key}'])        
        metricas = pd.concat([metricas, ste])

    rivais = copy.deepcopy(scores_rivais)

    nomes = {}
    nomes['TE']     = 'Target Encoding'
    nomes['MEst']   = 'M-Estimate Encoding'
    nomes['JSE']    = 'James-Stein Encoding'
    nomes['QuantE'] = 'Quantile Encoding'
    nomes['LOOE']   = 'Leave One Out Encoding'
    nomes['OE']     = 'Ordinal Encoding'

    for key, valor in rivais.items():
        for i in nomes.keys():
            if key.startswith(i):
                rival = pd.DataFrame(valor.values, columns=valor.columns, index=[f'{nomes[i]}{key[len(i):]}'])
                metricas = pd.concat([metricas, rival])

    metricas = metricas.reset_index()
    metricas = metricas.rename(columns={"index": "Encoder"})
    metricas = metricas.reset_index()
    metricas = metricas.rename(columns={"index": "Model"})
    model = [model_name] * len(metricas)
    metricas['Model'] = model
    
    return metricas

def printar_resultados(metricas_RBTE, metricas_rivais, model_name):

    medias_rbte, medias_rivais = medias_scores(metricas_RBTE, metricas_rivais)
    desvios_rbte, desvios_rivais = desvios_scores(metricas_RBTE, metricas_rivais)
    medias = metricas_to_DataFrame(medias_rbte, medias_rivais, model_name)
    desvios = metricas_to_DataFrame(desvios_rbte, desvios_rivais, model_name)

    return medias, desvios

def custom_cross_validation(data: pd.DataFrame, encoder, coluna_alvo:str, modelo, model_params = None):
    rmse = []
    mae = []

    k_folds = KFold(n_splits=10, random_state=721, shuffle=True) 

    for train, test in k_folds.split(data):        
        rbte_treino, rbte_teste = encode_rbte(data.loc[train], data.loc[test], encoder, coluna_alvo)

        rbte_treino_normal, rbte_teste_normal = normalizar(rbte_treino, rbte_teste, coluna_alvo)

        dataset_treino_X, dataset_treino_Y = data_split(rbte_treino_normal, coluna_alvo)
        dataset_teste_X,  dataset_teste_Y  = data_split(rbte_teste_normal, coluna_alvo)
        
        if model_params != None:
            model = modelo(**model_params)
        else:
            model = modelo()
            
        model.fit(dataset_treino_X, dataset_treino_Y)

        y_preds = model.predict(dataset_teste_X)

        rmse_teste = math.sqrt(mean_squared_error(dataset_teste_Y, y_preds))
        mae_teste  = mean_absolute_error(dataset_teste_Y, y_preds)
        
        rmse.append(rmse_teste)
        mae.append(mae_teste)    
        
    scores_teste = pd.DataFrame({'RMSE': rmse,'MAE': mae})

    return scores_teste

def impute_mean(df):
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    data = imputer.fit_transform(df)
    return data

def get_metricas(dataframe, STEncoder, EncodersRivais, modelo, coluna_alvo, model_params):
    scores_rbte = {}
    scores_rivais = {}

    for key, _ in STEncoder.items():
        scores_rbte[key] = custom_cross_validation(dataframe, STEncoder[key], coluna_alvo, modelo, model_params)

    for key, _ in EncodersRivais.items():
        scores_rivais[key] = custom_cross_validation(dataframe, EncodersRivais[key], coluna_alvo, modelo, model_params)

    return scores_rbte, scores_rivais