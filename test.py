
import pickle_compat
pickle_compat.patch()

import pickle
import pandas as pd
from sklearn.svm import SVC
from train import extract_features, normalize, perform_PCA, get_PCA


model = pickle.load(open( "model.pkl", "rb" ))

test_data = pd.read_csv('test.csv', header = None, low_memory = False).values.tolist()
F_test_df = extract_features(test_data)

#Normalize
F_test_df_normalized = normalize(F_test_df)

#Principal Component Analysis
def perform_PCA_for_test_data(dataset):
    pca = get_PCA() #assuming the pca is fit using train data during training
    transformed_dataset = pca.transform(dataset)
    return pd.DataFrame(transformed_dataset)

#X_test = perform_PCA(F_test_df_normalized)
X_test = perform_PCA_for_test_data(F_test_df_normalized)

#Predictions
y_predictions = model.predict(X_test)
y_predictions_df = pd.DataFrame(y_predictions)

y_predictions_df.to_csv('Results.csv', index = False, header = False)
