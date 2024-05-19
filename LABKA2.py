import numpy as np
from sklearn import preprocessing

input_data = np.array([[1.3, 3.9, 6.2],
                        [4.9, 2.2, -4.3],
                        [-2.6, 6.5, 4.1],
                        [-5.2, -3.4, -5.2]])

# Масштабування MinМax
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
print("\nМin max scaled data:\n", data_scaled_minmax)
