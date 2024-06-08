import pandas as pd
import category_encoders as ce
from sklearn.neighbors import KNeighborsRegressor
from common_scripts import STE, draft_utils
from datetime import datetime
import multiprocessing
from multiprocessing import Process

class Experiments:
    def __init__(self, airbnb_path, bike_sharing_path, co2_emissions_path, sp_air_path) -> None:
        self.airbnb_file        = airbnb_path
        self.bike_sharing_file  = bike_sharing_path
        self.co2_emissions_file = co2_emissions_path
        self.sp_air_file        = sp_air_path

        
    def get_encoders(self, smoothing):
        STEs = {}
        rivais = {}

        STEs["com_smooth"]       = STE.STEncoder(smooth=100, divide_by_smooth=True)
        STEs["com_smooth_0_50x"] = STE.STEncoder(smooth=smoothing[2], divide_by_smooth=True)
        STEs["com_smooth_2x"]    = STE.STEncoder(smooth=smoothing[3], divide_by_smooth=True)        

        rivais["TE"]                  = ce.TargetEncoder(smoothing=100)
        rivais["TE_smooth_0_50x"]     = ce.TargetEncoder(smoothing=smoothing[2])
        rivais["TE_smooth_2x"]        = ce.TargetEncoder(smoothing=smoothing[3])
        rivais["MEst"]                = ce.MEstimateEncoder(m=100, sigma= 0.05)
        rivais["MEst_smooth_0_50x"]   = ce.MEstimateEncoder(m=smoothing[2], sigma= 0.05)
        rivais["MEst_smooth_2x"]      = ce.MEstimateEncoder(m=smoothing[3], sigma= 0.05)
        rivais["JSE"]                 = ce.JamesSteinEncoder(sigma=0.05)
        rivais["QuantE"]              = ce.QuantileEncoder(m=100)
        rivais["QuantE_smooth_0_50x"] = ce.QuantileEncoder(m=smoothing[2])
        rivais["QuantE_smooth_2x"]    = ce.QuantileEncoder(m=smoothing[3])
        rivais["LOOE"]                = ce.LeaveOneOutEncoder(sigma=0.05)
        rivais["OE"]                  = ce.OrdinalEncoder()

        return STEs, rivais
        
    def save_results(self, medias, desvios, STEs_scores, scores_rivais, dataset_name):

        cv_STE = pd.DataFrame()
        cv_Rivais = pd.DataFrame()

        cv_STE = cv_STE.reset_index()
        cv_STE = cv_STE.rename(columns={"index": "Encoder"})

        cv_Rivais = cv_Rivais.reset_index()
        cv_Rivais = cv_Rivais.rename(columns={"index": "Encoder"})


        for key, value in STEs_scores.items():
            CV_results = value
            CV_results = CV_results.reset_index()
            CV_results = CV_results.rename(columns={"index": "Encoder"})
            encoder = [f'STE_{key}'] * len(CV_results)
            CV_results['Encoder'] = encoder
            cv_STE = pd.concat([cv_STE, CV_results])

        for key, value in scores_rivais.items():
            CV_results = value
            CV_results = CV_results.reset_index()
            CV_results = CV_results.rename(columns={"index": "Encoder"})
            encoder = [key] * len(CV_results)
            CV_results['Encoder'] = encoder
            cv_Rivais = pd.concat([cv_Rivais, CV_results])

        CVs = pd.concat([cv_STE, cv_Rivais])

        CVs.to_csv(f'resultados/CV_{dataset_name}.csv', index=False)

        medias.to_csv(f'resultados/{dataset_name}_means.csv', index=False)
        desvios.to_csv(f'resultados/{dataset_name}_stds.csv', index=False)

    def run_experiment(self, dataset_file: str, dataset_name: str, target_col: str):

        # Preprocess dataset
        dataframe = pd.read_csv(dataset_file)

        # Get smoothing values
        smoothing = draft_utils.get_smoothing_parameter(dataframe)

        STEs, rivais = self.get_encoders(smoothing=smoothing)

        # Run cross validated experiments on model
        STEs_scores, scores_rivais = draft_utils.get_metricas(dataframe=dataframe, \
                                                            STEncoder=STEs, \
                                                            EncodersRivais=rivais,\
                                                            modelo=KNeighborsRegressor, \
                                                            coluna_alvo=target_col, \
                                                            model_params={"n_jobs": -1} )
        
        # Get the results
        medias, desvios = draft_utils.printar_resultados(STEs_scores, scores_rivais, "KNN")

        self.save_results(medias, desvios, STEs_scores, scores_rivais, dataset_name)
    
    def airbnb(self):
        self.run_experiment(self.airbnb_file, "airbnb", "review_scores_rating")

    def bike_sharing(self):
        self.run_experiment(self.bike_sharing_file, "bike_sharing", "count")

    def co2_emission(self):
        self.run_experiment(self.co2_emissions_file, "co2_emissions", "CO2 Emissions(g/km)")

    def sp_air(self):
        self.run_experiment(self.sp_air_file, "sp_air", "MP10/(ug/m3)")

if __name__ == "__main__":
    print('Experiments initiated')

    experiments = Experiments("datasets/preprocessed/Airbnb_preprocessed.csv",\
                              "datasets/preprocessed/Bike_Sharing_preprocessed.csv",\
                              "datasets/preprocessed/co2_emissions_preprocessed.csv",\
                              "datasets/preprocessed/sp_air_preprocessed.csv"                                           
                             )


    airbnb_proc         = Process(target = experiments.airbnb)
    bike_sharing_proc   = Process(target = experiments.bike_sharing)
    co2_emission_proc   = Process(target = experiments.co2_emission)
    sp_air_proc         = Process(target = experiments.sp_air)

    airbnb_proc.start()
    bike_sharing_proc.start()
    co2_emission_proc.start()
    sp_air_proc.start()

    sp_air_proc.join()
    bike_sharing_proc.join()
    co2_emission_proc.join()
    airbnb_proc.join()

    print('Experiments finished')