import openml    

bike = openml.datasets.get_dataset(42712)
X, y, a, b = bike.get_data(dataset_format="dataframe")
X.to_csv(f"{bike.name}_openml_42712.csv", index=False)