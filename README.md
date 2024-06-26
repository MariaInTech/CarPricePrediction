# Predicting Car Prices with Neural Networks

In this project, we build a neural network model to predict the price of a car given a range of features such as the car’s manufacturer, levy, category, color, and more. We use a dataset of car prices from Kaggle to preprocess, train, and evaluate the model.

## Table of Contents
1. [Introduction](#introduction)
2. [Preparing the Data](#preparing-the-data)
3. [Model Architecture](#model-architecture)
4. [Results](#results)
5. [Conclusion](#conclusion)
6. [Installation and Usage](#installation-and-usage)
7. [Dependencies](#dependencies)
8. [License](#license)


## Introduction
This project aims to predict car prices using a neural network model. We employ a dataset from Kaggle, containing various features related to car specifications. The project involves data preprocessing, model training, and evaluation to achieve a high accuracy in price prediction.

## Preparing the Data
Before building the model, we prepared and cleaned the data:

1. **Data Import and Split:**
    - Imported the car prices dataset using pandas.
    - Split the dataset into features (X) and target (Y). The features are the columns used for analysis and prediction, and the target is the car price.

2. **Handling Categorical Features:**
    - Converted categorical features to numerical values using encoding techniques.
    - Used Python dictionaries and `pd.get_dummies()` to create new columns for each unique value in the categorical features.

3. **Visualization:**
    - Added plots to show the relationship between car prices and various features.

4. **Standardization:**
    - Standardized numerical features using scikit-learn’s `StandardScaler` to ensure the features have a mean of 0.

## Model Architecture
After preparing the data, we built the neural network model using Keras and TensorFlow:

1. **Model Definition:**
    - Created a function to build the model.
    - The model consists of:
        - Two hidden layers with 32 and 64 nodes, respectively.
        - An output layer with one node for predicting car prices.

2. **Compilation:**
    - Used the Adam optimizer and mean squared error (MSE) loss function.
    - Trained the model using the `fit` method with:
        - A training set with a validation split of 20%.
        - 100 epochs.

## Results
After training the model, we evaluated its performance on the testing set:

1. **Performance Metrics:**
    - Used the `r2_score` function from scikit-learn.
    - Achieved an R-squared score of 0.99, indicating a very high level of accuracy.

2. **Validation:**
    - Created a CSV file containing predicted values and actual car prices.
    - Observed that the predicted values were very close to the actual values, confirming the model's effectiveness.

## Conclusion
This project demonstrates the application of neural networks in predicting car prices with high accuracy. By carefully preprocessing the data and tuning the model, we achieved excellent performance. The results are promising for further exploration and improvement.

## Installation and Usage
To run this project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/car-price-prediction.git
    ```
2. Navigate to the project directory:
    ```bash
    cd car-price-prediction
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the data preprocessing script:
    ```bash
    python preprocess_data.py
    ```
5. Train the model:
    ```bash
    python train_model.py
    ```
6. Evaluate the model:
    ```bash
    python evaluate_model.py
    ```

## Dependencies
- Python 3.x
- pandas
- numpy
- scikit-learn
- tensorflow
- matplotlib

## License
This project is licensed under the MIT License - see the LICENSE file for details.

Feel free to contribute to this project by submitting issues or pull requests.
