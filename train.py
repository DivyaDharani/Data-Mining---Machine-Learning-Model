import pandas as pd
import warnings

from datetime import datetime, timedelta
import numpy as np
#import matplotlib.pyplot as plt
from numpy.fft import fft
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.tree import DecisionTreeClassifier

import pickle_compat
pickle_compat.patch()


warnings.filterwarnings("ignore")

def get_datasets(insulin_data_file_path, cgm_data_file_path, date_time_format):
    if date_time_format == '':
        date_time_format = '%m/%d/%Y %H:%M:%S'
    insulin_dataset_full = pd.read_csv(insulin_data_file_path, low_memory = False)
    insulin_data = insulin_dataset_full[['Date', 'Time', 'BWZ Carb Input (grams)']]
    cgm_data_set_full = pd.read_csv(cgm_data_file_path, low_memory = False)
    cgm_data = cgm_data_set_full[['Date', 'Time', 'Sensor Glucose (mg/dL)']]
    cgm_data.dropna(inplace = True)
    insulin_data['DateTime'] = pd.to_datetime(insulin_data['Date'] + ' ' + insulin_data['Time'], format = date_time_format)
    cgm_data['DateTime'] = pd.to_datetime(cgm_data['Date'] + " " + cgm_data['Time'], format = date_time_format)
    return insulin_data, cgm_data

def get_meal_start_times(insulin_dataset):
    insulin_data_carb = insulin_dataset[insulin_dataset['BWZ Carb Input (grams)'].notna() & insulin_dataset['BWZ Carb Input (grams)'] != 0]
    insulin_data_carb.rename({'DateTime' : 'MealStartDateTime'}, axis = 1, inplace = True)
    insulin_data_carb.sort_values(by = 'MealStartDateTime', inplace = True)
    meal_start_times = list(insulin_data_carb['MealStartDateTime'])
    return meal_start_times


def extract_meal_data(cgm_dataset, meal_start_times):
    valid_meal_start_times = []
    for i in range(len(meal_start_times)):
        timestamp = meal_start_times[i]
        if i > 0:
            previous_timestamp = meal_start_times[i-1]
            if previous_timestamp > timestamp - timedelta(hours = 0.5):
                continue

        if i < len(meal_start_times) - 1:
            next_timestamp = meal_start_times[i+1]
            if next_timestamp < timestamp + timedelta(hours = 2):
                continue

        valid_meal_start_times.append(timestamp)
    
    meal_data = []
    for meal_time in valid_meal_start_times:
        start_time = meal_time - timedelta(minutes = 30)
        end_time = meal_time + timedelta(hours = 2)
        filtered_data = cgm_dataset[(cgm_dataset['DateTime'] >= start_time) & (cgm_dataset['DateTime'] <= end_time)]
        if len(filtered_data) > 0:
            meal_data.append(list(filtered_data['Sensor Glucose (mg/dL)'].values))
    #meal_data_df = pd.DataFrame(meal_data)
    return meal_data

def extract_no_meal_data(insulin_dataset, cgm_dataset, meal_start_times):
    start_times_to_consider = []

    start_times_to_consider.append(min(insulin_dataset['DateTime']) - timedelta(hours = 2)) 
    #below steps add +2 assuming the list has meal start times (we should consider no meal time only after 2 hours after eating)
    #that is why subtracting -2 hours here so that +2 later will be having the same time

    start_times_to_consider.extend(meal_start_times)
    start_times_to_consider.append(max(insulin_dataset['DateTime']))
    
    valid_no_meal_start_times = []
    for i in range(1, len(start_times_to_consider)):
        start = start_times_to_consider[i-1] + timedelta(hours = 2)
        currMealTime = start_times_to_consider[i]
        while (start + timedelta(hours = 2)) <= currMealTime:
            valid_no_meal_start_times.append(start)
            start += timedelta(hours = 2)
    
    no_meal_data = []
    for start_time in valid_no_meal_start_times:
        end_time = start_time + timedelta(hours = 2)
        filtered_data = cgm_dataset[(cgm_dataset['DateTime'] >= start_time) & (cgm_dataset['DateTime'] <= end_time)]
        if len(filtered_data) > 0:
            no_meal_data.append(list(filtered_data['Sensor Glucose (mg/dL)'].values))
    #no_meal_data_df = pd.DataFrame(no_meal_data)
    return no_meal_data

def compute_slope_features(datarow): 
    """
    datarow: list of values
    This method computes differential values for every 3 consecutive points g1, g2, g3 at t1, t2, t3 in this way: 
    slope = (g1+g3-2g2)/(t3-t1)
    Then, at zero-crossing indices, |max-min slope| is calculated and top 3 such values are picked
    """
    slopes = []
    for i in range(len(datarow)-2):
        slopes.append((datarow[i] + datarow[i+2] - 2 * datarow[i+1]) / ((i+2-i) * 5.0)) #one reading per 5 minutes
    
    #plt.plot(slopes)
    #plt.axhline(y=0, color='r')
    
    zero_crossing_indices = np.where(np.diff(np.sign(slopes)))[0]
    #zero-crossing |max-min slope| with indices
    zero_crossing_delta = [(index, abs(slopes[index+1]-slopes[index])) for index in zero_crossing_indices]
    zero_crossing_delta.sort(key = lambda x: x[1], reverse = True)
    return zero_crossing_delta[:3] #top 3 values

def frequency_domain_features(datarow): #computes fft and returns the 2nd, 3rd, and 4th max freq indices
    frequencies = fft(datarow)
    #2nd, 3rd, and 4th max freq indices
    top_frequency_indices = np.argsort(frequencies)[::-1][1:4]
    return top_frequency_indices.tolist()

def extract_features(dataset): #dataset can be list of list values
    cgmMaxAndMinDiff = []
    cgmMaxAndMinTimeDiff = []
    slope_delta_1 = []
    slope_delta_1_loc = []
    slope_delta_2 = []
    slope_delta_2_loc = []
    slope_delta_3 = []
    slope_delta_3_loc = []
    slope_delta_features = [slope_delta_1, slope_delta_2, slope_delta_3]
    slope_delta_loc_features = [slope_delta_1_loc, slope_delta_2_loc, slope_delta_3_loc]
    fft_2 = []
    fft_3 = []
    fft_4 = []
    fft_features = [fft_2, fft_3, fft_4]

    for datarow in dataset:
        maxVal = max(datarow)
        minVal = min(datarow)
        #feature 1
        cgmMaxAndMinDiff.append(maxVal - minVal)

        #features 2-7
        slope_feature_tuples = compute_slope_features(datarow) #(l1, m1), (l2, m2), (l3, m3)
        for i in range(3): 
            if i < len(slope_feature_tuples):
                slope_delta_features[i].append(slope_feature_tuples[i][1])
                slope_delta_loc_features[i].append(slope_feature_tuples[i][0])
            else:
                slope_delta_features[i].append(None)
                slope_delta_loc_features[i].append(None)

        #feature 8
        cgmMaxAndMinTimeDiff.append((datarow.index(maxVal) - datarow.index(minVal)) * 5) #one reading per 5 minutes

        #features 9-11
        top_frequencies = frequency_domain_features(datarow)
        for i in range(3):
            if i < len(top_frequencies):
                fft_features[i].append(top_frequencies[i])
            else:
                fft_features[i].append(None)

    result_df = pd.DataFrame()
    result_df['CGM_Max_Min_Diff'] = cgmMaxAndMinDiff
    result_df['slope_delta_1'] = slope_delta_1
    result_df['slope_delta_1_loc'] = slope_delta_1_loc
    result_df['slope_delta_2'] = slope_delta_2
    result_df['slope_delta_2_loc'] = slope_delta_2_loc
    result_df['slope_delta_3'] = slope_delta_3
    result_df['slope_delta_3_loc'] = slope_delta_3_loc
    result_df['CGM_Max_Min_Time_Diff'] = cgmMaxAndMinTimeDiff
    result_df['fft_2'] = fft_2
    result_df['fft_3'] = fft_3
    result_df['fft_4'] = fft_4
    return result_df

def normalize(df):
    #return (df - df.mean())/(df.max() - df.min())
    return (df - df.min())/((df.max() - df.min()) * 1.0)

pca = PCA(n_components = 8)
def get_PCA():
    return pca

def perform_PCA(dataset):
    pca = get_PCA()
    pca.fit(dataset)
    transformed_dataset = pca.transform(dataset)
    #plt.plot(pca.explained_variance_)
    return pd.DataFrame(transformed_dataset)

def get_final_meal_and_no_meal_datasets(insulin_data_file_path, cgm_data_file_path, date_time_format):
    insulin_dataset, cgm_dataset = get_datasets(insulin_data_file_path, cgm_data_file_path, date_time_format)
    meal_start_times = get_meal_start_times(insulin_dataset)
    meal_data = extract_meal_data(cgm_dataset, meal_start_times)
    no_meal_data = extract_no_meal_data(insulin_dataset, cgm_dataset, meal_start_times)
    F_meal_data_df = extract_features(meal_data)
    F_no_meal_data_df = extract_features(no_meal_data)
    F_meal_data_df.dropna(inplace = True)
    F_no_meal_data_df.dropna(inplace = True)
    return F_meal_data_df, F_no_meal_data_df


def svm_classifier(X_train, y_train):
    svm_model = SVC(gamma = 'scale')
    svm_model.fit(X_train, y_train)
    return svm_model

def decision_tree_classifier(X_train, y_train):
    classifier = DecisionTreeClassifier(criterion = 'entropy')
    classifier.fit(X_train,y_train)
    return classifier

#Final 
#Testing --------
F_meal_df_1, F_no_meal_df_1 = get_final_meal_and_no_meal_datasets('InsulinData.csv', 'CGMData.csv', '%m/%d/%Y %H:%M:%S')
#F_meal_df_2, F_no_meal_df_2 = get_final_meal_and_no_meal_datasets('Insulin_patient2.csv', 'CGM_patient2.csv', '%d-%m-%Y %H:%M:%S')

#F_meal_df = pd.concat([F_meal_df_1, F_meal_df_2])
#F_no_meal_df = pd.concat([F_no_meal_df_1, F_no_meal_df_2])

F_meal_df = F_meal_df_1
F_no_meal_df = F_no_meal_df_1


#Assigning class labels:
F_meal_df['Class'] = 1
F_no_meal_df['Class'] = 0

#Concatenating meal and no-meal data:
F_data_df = pd.concat([F_meal_df, F_no_meal_df])

#Correcting the indices
F_data_df = F_data_df.reset_index().drop(columns = 'index')

#Taking the class lables out
class_labels = F_data_df['Class']
F_data_df.drop(columns = 'Class', inplace = True)

#Normalize
F_data_df_normalized = normalize(F_data_df)

#Principal Component Analysis
X_train = perform_PCA(F_data_df_normalized)

y_train = class_labels

model = svm_classifier(X_train, y_train)

pickle.dump(model, open('model.pkl', 'wb'))