import tensorflow as tf
import pandas as pd
import tensorflow.keras as keras

dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

colnums = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(dataset_path,names=colnums,na_values='?',comment="\t",sep=" ",skipinitialspace=True)

dataset =raw_dataset.copy()

print(dataset.head())