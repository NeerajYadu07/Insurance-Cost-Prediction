# Medical Insurance Cost Prediction with Linear Regression

## Overview

This repository contains a Machine Learning (ML) model for predicting medical insurance costs using Linear Regression. The model is trained on a dataset containing information about individuals, such as age, BMI, number of children, smoking status, and region, to predict their medical insurance costs.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Dependencies](#dependencies)
4. [Model Training](#model-training)
5. [Evaluation](#evaluation)
6. [Results](#results)

## Introduction

The goal of this project is to develop a predictive model that can estimate medical insurance costs for individuals based on various factors. Linear Regression is used as the chosen machine learning algorithm due to its simplicity and interpretability.

## Dataset

The dataset used for training and testing the model is included in the directory. The dataset contains the following columns:

- Age: Age of the individual.
- BMI: Body Mass Index of the individual.
- Children: Number of children/dependents covered by the insurance.
- Smoker: Smoking status of the individual (0 for non-smoker, 1 for smoker).
- Region: Geographic region of the individual.
- Charges: Medical insurance charges for the individual (target variable).

## Dependencies

Ensure you have the following dependencies installed before running the code:

- Python (>=3.6)
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Model Training
The model training process is documented in the Medical_Insurance_Cost_Prediction.ipynb Jupyter notebook. This notebook explains how the dataset is preprocessed, how the linear regression model is trained, and how the model is evaluated.

## Evaluation
The model's performance is evaluated using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared. These metrics provide insights into how well the model generalizes to new, unseen data.

## Results
The results of the model, including evaluation metrics and visualizations, are presented in the Jupyter Notebook. This information helps users understand the model's predictive capabilities and potential areas for improvement.
