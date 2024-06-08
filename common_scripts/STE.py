import pandas as pd
import random
import copy
import numpy as np

class STEncoder:
    def __init__(self, cols = None, smooth = 100.0, divide_by_smooth = True):
        self.cols_to_encode = cols
        self.catecoric_values = {}
        self.global_mean = None
        self.global_deviation = None
        self.cols_means = {}
        self.cols_deviations = {}
        self.data = None
        self.falhas = {}
        self.catecoric_values_count = {}
        self.pesos = {}
        self.smooth = smooth
        self.divide_by_smooth = divide_by_smooth

        # self.encoded_data = None

    def fit(self, dataframe:pd.DataFrame, target_col: pd.DataFrame):
        if self.cols_to_encode == None:
            self.cols_to_encode = self.__get_categoric_columns(dataframe)

        self.catecoric_values = self.__get_categoric_values(dataframe, self.cols_to_encode)

        self.global_mean, self.global_deviation = self.__get_global_mean_and_std(target_col)

        self.data = self.__get_data_to_encode(dataframe, self.cols_to_encode, target_col.name)

        self.__get_weights()

        for coluna in self.cols_to_encode:
            self.falhas[coluna] = 0


    def fit_transform(self, dataframe:pd.DataFrame, target_col: pd.DataFrame):
        self.fit(dataframe, target_col)
        encoded_data = self.transform(dataframe)

        return encoded_data#, self.cols_means, self.cols_deviations


    def transform(self, dataframe:pd.DataFrame):
        data_to_encode = copy.deepcopy(dataframe)

        for coluna in self.cols_to_encode:
            data_to_encode[coluna] = data_to_encode[coluna].apply(lambda x: self.__encode(x,self.data[coluna][0], self.data[coluna][1], coluna, self.catecoric_values))

        return data_to_encode#, self.cols_means, self.cols_deviations

    def __get_weights(self):
        # print("__get_weights")
        for chave, valor in self.catecoric_values_count.items():
            self.pesos[chave] = {}
            # print(valor)
            for key, _ in valor.items():
                # print(key)
                self.pesos[chave][key] = self.catecoric_values_count[chave][key] / (self.catecoric_values_count[chave][key] + self.smooth)
            # print(valor[0].index)
            # for index in valor[0].index:

    def get_weights(self):
        return self.pesos

    def __get_categoric_columns(self, dataframe:pd.DataFrame):
        cols = dataframe.columns
        num_cols = dataframe._get_numeric_data().columns

        categoric_cols = list(set(cols) - set(num_cols))

        return categoric_cols

    def __get_categoric_values(self, dataframe:pd.DataFrame, columns):
        valores_categoricos = {}

        for coluna in columns:
            valores = dataframe[coluna].unique()
            valores_categoricos[coluna] = valores
            self.catecoric_values_count[coluna] = {}
            self.catecoric_values_count[coluna] = dataframe[coluna].value_counts()

        return valores_categoricos

    def __get_global_mean_and_std(self, target_col: pd.DataFrame):
        media  = target_col.mean(0)
        desvio = target_col.std(0)

        return media, desvio

    def __get_mean_and_std(self, dataframe:pd.DataFrame, categoric_cols, target_col):
        # print("__get_mean_and_std")
        valores = {}
        medias = {}
        desvios = {}
        for coluna in categoric_cols:
            valores[coluna] = dataframe.groupby(coluna)[target_col].describe()
            # print(coluna)
            # print(valores[coluna])
            # print()

        for chave, valor in valores.items():
            # print(valor['mean'])
            # print(chave)
            medias[chave] = valor['mean']
            desvios[chave] = valor['std']

        return medias, desvios

    def __get_data_to_encode(self, dataframe:pd.DataFrame, categoric_cols, target_col):
        data = {}
        self.cols_means, self.cols_deviations = self.__get_mean_and_std(dataframe, categoric_cols, target_col)
        # print()
        # print("__get_data_to_encode")

        # print(self.cols_means.items())
        for chave, valor in self.cols_means.items():
            data[chave] = []

        for chave, valor in data.items():
            data[chave].append(self.cols_means[chave])
            data[chave].append(self.cols_deviations[chave])

        return data

    def __encode(self, variavel,media, desvio, coluna, categoric_values):

        if variavel in media.index:
            if np.isnan(desvio.loc[variavel]):

                b = ( (self.pesos[coluna][variavel] * media.loc[variavel]) + (1 - self.pesos[coluna][variavel]) * self.global_mean)
                return b
            else:                
                mean_encoded = ((self.pesos[coluna][variavel] * media.loc[variavel]) + (1 - self.pesos[coluna][variavel]) * self.global_mean)

                desvio_Xi = ((self.pesos[coluna][variavel] * desvio.loc[variavel]) + (1 - self.pesos[coluna][variavel]) * self.global_deviation)
                
                a = mean_encoded + random.uniform(-desvio_Xi, desvio_Xi)
                
                return a

        else:
            self.falhas[coluna] += 1
            if np.isnan(self.global_deviation):
                return self.global_mean
            else:
                return self.global_mean + random.uniform(-self.global_deviation, self.global_deviation)

    def get_falhas(self):
        return self.falhas

    def get_categoric_values_count(self):
        return self.catecoric_values_count

    def get_data(self):
        return self.data
