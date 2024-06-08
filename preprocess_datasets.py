import pandas as pd
from copy import deepcopy
import os
import sys
from common_scripts import draft_utils

class Datasets_Preprocessing:
    def __init__(self, airbnb_path, bike_sharing_path, co2_emissions_path, sp_air_path, data_count) -> None:
        self.airbnb_file        = airbnb_path
        self.bike_sharing_file  = bike_sharing_path
        self.co2_emissions_file = co2_emissions_path
        self.sp_air_file        = sp_air_path
        self.datasets_read_path = "/datasets/raw/"
        self.datasets_save_path = "datasets/preprocessed/"
        self.data_count         = data_count

    def preprocess_airbnb(self) -> None:
        df = pd.read_csv(self.airbnb_file)
        df.drop('thumbnail_url', axis=1, inplace=True)
        
        if(self.data_count != None):
            df = df.iloc[:self.data_count, :]

        df2 = deepcopy(df)
        df2 = df2.dropna(subset="review_scores_rating")
        df2 = df2.dropna(subset="host_response_rate")
        df2 = df2.dropna(subset="neighbourhood")
        df2 = df2.dropna(subset="zipcode")
        df2 = df2.dropna(subset=["first_review", "last_review"])
        df2 = df2.reset_index(drop=True)
        

        imputados = draft_utils.impute_mean(df2[['bathrooms', 'bedrooms', 'beds']])
        imputados = pd.DataFrame(imputados, columns=['bathrooms', 'bedrooms', 'beds'])
        df2[['bathrooms', 'bedrooms', 'beds']] = imputados
        del imputados
        df2.isna().sum()


        coluna_alvo = 'review_scores_rating'
        alvo = df2[coluna_alvo]
        df2.drop('review_scores_rating', axis=1, inplace=True)
        dataframe = pd.concat([df2, alvo], axis=1)


        def string_to_bool(x):
            if x == "t" :
                return True
            else:
                return False

        def remove_percentage_sign(x):
            new_string = x.replace("%", "")
            return new_string
            
        df = {}

        df["host_identity_verified"] = dataframe["host_identity_verified"].apply(string_to_bool)
        df["host_has_profile_pic"] = dataframe["host_has_profile_pic"].apply(string_to_bool)
        df["instant_bookable"] = dataframe["instant_bookable"].apply(string_to_bool)
        df["host_response_rate"] = dataframe["host_response_rate"].apply(remove_percentage_sign)
        df_exemplo = pd.DataFrame.from_dict(df)

        dataframe[["host_identity_verified", "host_has_profile_pic", "instant_bookable", "host_response_rate"]] = df_exemplo[["host_identity_verified", "host_has_profile_pic", "instant_bookable", "host_response_rate"]]

        dataframe.to_csv(f"{self.datasets_save_path}Airbnb_preprocessed.csv", index=False)

    def preprocess_bike_sharing(self) -> None:
        dataframe = pd.read_csv(self.bike_sharing_file, encoding='ISO-8859-1')

        if(self.data_count != None):
            dataframe = dataframe.iloc[:self.data_count, :]

        dataframe.to_csv(f'{self.datasets_save_path}Bike_Sharing_preprocessed.csv', index=False)

    def preprocess_co2_emissions(self) -> None:
        df = pd.read_csv(self.co2_emissions_file)
    
        df['Year'] = df['Year'].astype(str)

        if(self.data_count != None):
            df = df.iloc[:self.data_count, :]

        dataframe = df.drop(['Fuel Consumption City (L/100 km)', 'Fuel Consumption Hwy (L/100 km)', 'Fuel Consumption Comb (L/100 km)', 'Fuel Consumption Comb (mpg)', 'Engine Size(L)'], axis=1)

        dataframe.to_csv(f"{self.datasets_save_path}co2_emissions_preprocessed.csv", index=False)

    def preprocess_sp_air(self) -> None:
        df = pd.read_csv(self.sp_air_file)

        if(self.data_count != None):
            df = df.iloc[:self.data_count, :]

        df = df.loc[df['Poluente'] == "MP10"]
        df = df.reset_index(drop=True)
        df = df.drop(["Unnamed: 0", "Poluente", "Unidade"], axis=1)
        df.rename(columns={"Valor": "MP10/(ug/m3)"}, inplace=True)
        coluna_alvo = "MP10/(ug/m3)"
        alvo = df[coluna_alvo]
        df.drop(coluna_alvo, axis=1, inplace=True)
        dataframe = pd.concat([df, alvo], axis=1)

        dataframe.to_csv(f"{self.datasets_save_path}sp_air_preprocessed.csv", index=False)

if __name__ == '__main__':
    preprocessing = Datasets_Preprocessing("datasets/raw/Airbnb_Data.csv",\
                                           "datasets/raw/Bike_Sharing_Demand_openml_42712.csv",\
                                           "datasets/raw/Fuel_Consumption_2000-2022.csv",\
                                           "datasets/raw/SP_poluicao_dados.csv",\
                                           200
                                          )

    preprocessing.preprocess_airbnb()
    preprocessing.preprocess_bike_sharing()
    preprocessing.preprocess_co2_emissions()
    preprocessing.preprocess_sp_air()

    print("Datasets preprocessed")