# Distinguishing between meal and no-meal time-series data in Artificial Pancreas System

The purpose of this project is to use the sensor data in the Artificial Pancreas System to assess whether a person has eaten a meal or not in a particular time period. 

## Datasets
* Continuous Glucose Sensor data (CGMData.csv)
* Insulin pump data (InsulinData.csv)


## Extraction of Meal data
Start of a meal can be obtained from the carbohydrate input data from InsulinData.csv. Meal data comprises a 2 hr 30 min stretch of CGM data from the start time. 

## Extraction of No-Meal data
No meal data comprises 2 hrs of raw data that does not have meal intake.

## Feature Extraction
Extracted slope and frequency domain features from the curated meal and no-meal data.

## Dimensionality Reduction:
Used Principal Component Analysis

## Machine Learning Algorithms
Used Decision Tree and Support Vector Machine (SVM) classifiers, and evaluated the models using k-fold cross validation technique. 




