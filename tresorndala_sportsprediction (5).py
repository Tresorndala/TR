# -*- coding: utf-8 -*-
"""TRESORNDALA_sportsprediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1GZ98gAKwxUmSZprdixeVp1kxt8uFxxYP

Step 1 import the first libraries i am needing for my project and my initial data.
"""

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

"""Step 2:**Loading The data that i put in my drive for colab and allowing
permission**
"""

# Mount Google Drive if using Google Colab
from google.colab import drive
drive.mount('/content/drive')

# Specify the file paths for training and testing data
file_path = '/content/drive/My Drive/male_players (legacy).csv'

"""Data loading

"""

# Load the CSV files into DataFrames
data = pd.read_csv(file_path)

"""Data inspection"""

# Display the first few rows of the DataFrames to verify they loaded correctly
data.head()

"""Step 3:**Describing** deeply the data for deep **inference**

"""

# Describe the data
print(data.describe())

"""investigate if the data has missing values"""

data.info()
data.isnull()

"""it seems from the top that the missing values are not present but we shall continue with investication and still use the imputer to learn the data and check even within the categorical to fill mode

step4:Importing simpleImputer to learn the missing values and data before filling them by the median or mean strategy.
"""

from sklearn.impute import SimpleImputer

# Check for missing values

print(data.isna().sum())


# Impute missing values
for column in data.columns:
    if data[column].dtype == 'object':
        data[column].fillna(data[column].mode()[0], inplace=True)
    else:
        imputer = SimpleImputer(strategy='median') if data[column].median() < data[column].mean() else SimpleImputer(strategy='mean')
        data[[column]] = imputer.fit_transform(data[[column]])


print(data.isna().sum())



"""step 5 checking the data if they are categorical we select them and put them in one variable and the same for the numerical because of good treatment of data  and doing  Data Visualization and Analysis"""

# Check categorical columns
cat_cols = data.select_dtypes(include=['object']).columns
print("Categorical columns:")
print(cat_cols)

# Check numerical columns
num_cols = data.select_dtypes(include=['int64', 'float64']).columns

"""Separate categorical and numerical data"""

categorical_data = data[cat_cols].copy()  # Using copy() to avoid SettingWithCopyWarning
numerical_data = data[num_cols]

"""investicate each category"""

categorical_data.info()

numerical_data.info()

"""Data Visualization and Analysis"""

#  Boxplot of numerical columns
plt.figure(figsize=(10, 6))
sns.boxplot(data=numerical_data, orient='h')
plt.title('Boxplot of Numerical Columns')
plt.xlabel('Values')
plt.show()

""" Correlation heatmap of numerical columns"""

#  Correlation heatmap of numerical columns
correlation_matrix = numerical_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Numerical Columns')
plt.show()

"""Step 6:Enconding already the categorical data that seems to be relevant for compatible treatment by our objective of predicting the overall performance of the player,as we know that a player who plays to the right ,there is a big influence if he comes to change his position of prefered foot in terms of performance,we choose it as the most important categorical variable to encode."""

# Encode categorical data
categorical_data_encoded = pd.get_dummies(categorical_data['preferred_foot']).astype(int)

"""Converting the one hot encoded in sparse because it is the recommended form"""

# Convert the one-hot encoded DataFrame to sparse
categorical_data_encoded = categorical_data_encoded.astype(pd.SparseDtype("int", 0))
categorical_data_encoded



"""step 7 Separate output numerical data (target variable) or dependent variable and input numerical data.

target_variable = 'overall' symbolising the overall perforfance of each player.
"""

# Separate output numerical data (target variable) and input numerical data
target_variable = 'overall'
y_numerical = numerical_data[target_variable]
X_numerical = numerical_data.drop(columns=[target_variable])

print("Input numerical data:")
print(X_numerical.head())
print("Output numerical data (target variable):")
print(y_numerical.head())



"""Step 8 Defining the correlation function to find the inputs variables that are are highly correlated between themselses and remove them. in fact for example if the input are correlated with 0.9 that means that when the model will be used to prdict it will be exposed to confusion not knowing to differentiate what will affect negatively the prediction.

Removing highly correlated input data is essential in machine learning to improve model performance and prevent overfitting. Highly correlated features provide redundant information, which can confuse learning algorithms and lead to poor generalization to new data (Brownlee, 2020). Functions like correlation(X_numerical, threshold) help identify and remove these features, streamlining the dataset and enhancing model accuracy (Raschka & Mirjalili, 2017). Additionally, reducing feature correlation decreases the variance of model coefficients, making the model more robust to changes in the training data (Hastie, Tibshirani, & Friedman, 2009). Overall, this practice results in more efficient training and better model interpretability.
"""

# Define the correlation function to remove highly correlated inputs variables between themselves to avoid confusion of dupplicates in the prediction.
def correlation(X_numerical, threshold):
    col_corr = set()
    corr_matrix = pd.DataFrame(X_numerical).corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr



"""step9 Define the function to remove weakly correlated features with the output variable

Defining a function to remove weakly correlated features with the output variable is crucial in machine learning for enhancing model performance and interpretability. Weakly correlated features contribute less information to the prediction of the target variable and can introduce noise into the model (Hastie, Tibshirani, & Friedman, 2009). By identifying and removing these features, as implemented in the function weak_correlation_with_target(X_numerical, y_numerical, threshold), we improve the model’s ability to focus on the most influential predictors (Raschka & Mirjalili, 2017). This process helps mitigate the risk of overfitting and ensures that the model learns meaningful patterns from the data (Brownlee, 2020). Ultimately, eliminating weakly correlated features leads to more robust and accurate models, as they are less likely to be influenced by irrelevant or noisy data points.
"""

# Define the function to remove weakly correlated features with the output variable
def weak_correlation_with_target(X_numerical, y_numerical, threshold):
    weak_corr_features = set()
    for col in range(X_numerical.shape[1]):
        correlation = abs(pd.Series(X_numerical.iloc[:, col]).corr(pd.Series(y_numerical)))
        if correlation < threshold:
            weak_corr_features.add(X_numerical.columns[col])
    return weak_corr_features

"""#Step 10 Apply the correlation function to identify highly correlated features"""

# Apply the correlation function to identify highly correlated features
cor_features = correlation(X_numerical, 0.65)
print(f"Highly correlated input features to remove and keep one to avoid confusion due to dupplicates: {cor_features}")

"""step 11 Remove the highly correlated features from the dataset"""

# Remove the highly correlated features from the dataset
X_numerical.drop(columns=cor_features, inplace=True)

print("Input numerical data after removing highly correlated features:")
print(X_numerical.head())

"""step 12 identify weakly correlated features with the target"""

# Apply the weak correlation function to identify weakly correlated features with the target
weak_corr_features = weak_correlation_with_target(X_numerical, y_numerical, 0.45)
print(f"Weakly correlated features to remove: {weak_corr_features}")

# Remove the weakly correlated features from the dataset
X_numerical.drop(columns=weak_corr_features, inplace=True)

print("Input numerical data after removing weakly correlated features:")
print(X_numerical.head())

X_numerical

"""step 13 Combine numerical and categorical data"""

# Combine numerical and categorical data
combined_data = pd.concat([X_numerical, categorical_data_encoded], axis=1)

# Convert combined data to sparse format
combined_data_sparse = combined_data.astype(pd.SparseDtype("float", 0))

"""# step 14 Combine numerical and categorical data and remove the data that passed all the test but based on Pairplot of All Numerical Columns in X_numerical', y=1.02)  they are not so important to enter in the training."""

import seaborn as sns
import matplotlib.pyplot as plt

# Create a pairplot for all numerical columns in X_numerical
sns.pairplot(X_numerical, diag_kind='hist', plot_kws={'alpha': 0.7})
plt.suptitle('Pairplot of All Numerical Columns in X_numerical', y=1.02)  # Adjust title position
plt.show()

# Convert combined data to sparse format
combined_data_sparse = combined_data.astype(pd.SparseDtype("float", 0))
combined_data_sparse
# Drop the specified columns
columns_to_drop = ['player_id', 'fifa_update','value_eur','preferred_foot']
combined_data_sparse = combined_data_sparse.drop(columns=columns_to_drop, errors='ignore')
combined_data_sparse

print(y_numerical)

import matplotlib.pyplot as plt

# Plot a histogram for the 'age' column
plt.figure(figsize=(10, 6))
plt.hist(combined_data_sparse['age'], bins=30, edgecolor='k', alpha=0.7)
plt.title('Distribution of Player Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt

# Scatter plot for 'shooting' vs 'overall'
plt.figure(figsize=(10, 6))
plt.scatter(data['shooting'], data['overall'], alpha=0.7, edgecolor='k')
plt.title('Shooting vs Overall Rating')
plt.xlabel('Shooting')
plt.ylabel('Overall Rating')
plt.grid(True)
plt.show()

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

"""# step 15 Convert to scipy sparse matrix"""

import pickle as pkl
import joblib
# Convert to scipy sparse matrix
from scipy.sparse import csr_matrix
combined_data_sparse_matrix = csr_matrix(combined_data_sparse.sparse.to_coo())

# Standardize features
scaler = StandardScaler(with_mean=False)  # with_mean=False is required for sparse data
X_scaled = scaler.fit_transform(combined_data_sparse_matrix)

# Save the scaler to Google Drive
scaler_path = '/content/drive/My Drive/scaler.pkl'
joblib.dump(scaler, scaler_path)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_numerical, test_size=0.2, random_state=42)

# Data Visualization: Correlation matrix of the training set after removing highly and weakly correlated features
plt.figure(figsize=(12, 8))
sns.heatmap(pd.DataFrame(X_train.toarray()).corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix (Training Data)')
plt.show()

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

"""step 16 Initialize models"""

lr = LinearRegression()
rf = RandomForestRegressor(random_state=42)
gb = GradientBoostingRegressor(random_state=42)

import pickle as pkl
import joblib

"""step 17:The train_and_evaluate_model function automates the process of training and evaluating machine learning models, facilitating model selection and hyperparameter tuning. It constructs a pipeline that includes data scaling and a specified regression model. Depending on the model_name provided (e.g., 'RandomForest' or 'GradientBoosting'), the function defines a grid of hyperparameters to search over using GridSearchCV, a methodical search technique. This grid includes parameters such as number of estimators, maximum depth, and minimum samples per split, tailored to each model type to optimize performance. The function performs cross-validation to evaluate models based on mean squared error (MSE) and mean absolute error (MAE). After identifying the best estimator based on cross-validation results, it trains this estimator on the training data, evaluates its performance on test data, and stores the trained model for future use. This streamlined approach ensures robust model training and selection, enhancing predictive accuracy and enabling informed decision-making in machine learning applications."""

from sklearn.model_selection import GridSearchCV

def train_and_evaluate_model(model, model_name):
    pipeline = Pipeline(steps=[
        ('scaler', scaler),
        ('regressor', model)
    ])

    # Define parameter grid based on the model
    if model_name == 'RandomForest':
        param_grid = {
            'regressor__n_estimators': [100, 200, 300],
            'regressor__max_depth': [None, 10, 20, 30],
            'regressor__min_samples_split': [2, 5, 10],
            'regressor__min_samples_leaf': [1, 2, 4],
            'regressor__bootstrap': [True, False]
        }
    elif model_name == 'GradientBoosting':
        param_grid = {
            'regressor__n_estimators': [100, 200, 300],
            'regressor__learning_rate': [0.01, 0.1, 0.2],
            'regressor__max_depth': [3, 5, 7],
            'regressor__min_samples_split': [2, 5, 10],
            'regressor__min_samples_leaf': [1, 2, 4],
            'regressor__subsample': [0.8, 0.9, 1.0]
        }
    else:
        param_grid = {}

    # Use GridSearchCV for exhaustive search
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_estimator = grid_search.best_estimator_
    cv_rmse = np.sqrt(-grid_search.best_score_)
    cv_mae = -cross_val_score(best_estimator, X_train, y_train, cv=5, scoring='neg_mean_absolute_error').mean()

    print(f"{model_name} CV RMSE: {cv_rmse}, CV MAE: {cv_mae}")

    best_estimator.fit(X_train, y_train)
    y_pred = best_estimator.predict(X_test)

    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    print(f"{model_name} Test RMSE: {test_rmse}")
    print(f"{model_name} Test MAE: {test_mae}")
    print(f"{model_name} Test R^2: {test_r2}")

    joblib.dump(best_estimator, f'/content/drive/My Drive/{model_name}.pkl')

    return best_estimator, test_rmse, test_mae

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV

""" Training  and evaluate models"""

# Train and evaluate models
models = {'LinearRegression': lr, 'RandomForest': rf, 'GradientBoosting': gb}
best_model = None
best_rmse = float('inf')
best_mae = float('inf')

for model_name, model in models.items():
    trained_model, test_rmse, test_mae = train_and_evaluate_model(model, model_name)
    if test_rmse < best_rmse:
        best_model = trained_model
        best_rmse = test_rmse
        best_mae = test_mae

print(f"Best model: {best_model}")
print(f"Best model RMSE: {best_rmse}")
print(f"Best model MAE: {best_mae}")

"""/usr/local/lib/python3.10/dist-packages/sklearn/model_selection/.
  
/usr/local/lib/python3.10/dist-packages/joblib/externals/loky/backend/fork_exec.py:
  pid = os.fork()
LinearRegression CV RMSE: 2.1395360536389116, CV MAE: 1.6619334486768227
LinearRegression Test RMSE: 2.141846380281537
LinearRegression Test MAE: 1.6643978910633335
LinearRegression Test R^2: 0.9074419182163047
/usr/local/lib/python3.10/dist-packages/sklearn/model_selection/

RandomForest CV RMSE: 1.4714582234676465, CV MAE: 0.9659791211580465
RandomForest Test RMSE: 1.4578246824553418
RandomForest Test MAE: 0.9585622113709193
RandomForest Test R^2: 0.9571206388642849
/usr/local/lib/python3.10/dist-packages/sklearn/model_selection/

/usr/local/lib/python3.10/dist-packages/joblib/externals/loky/backend/fork_exec.py:38: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
  pid = os.fork()
GradientBoosting CV RMSE: 1.4188752175355834, CV MAE: 0.969227288041633
GradientBoosting Test RMSE: 1.4153550451048034
GradientBoosting Test MAE: 0.968687836618711
GradientBoosting Test R^2: 0.9595825878757873
Best model: Pipeline(steps=[('scaler', StandardScaler(with_mean=False)),
                ('regressor',
                 GradientBoostingRegressor(max_depth=5, random_state=42))])

Best model RMSE: 1.4153550451048034
Best model MAE: 0.968687836618711





**Testing result explanation**

The training and evaluation of various machine learning models, including Linear Regression, Random Forest, and Gradient Boosting, demonstrated that Gradient Boosting achieved superior performance. With a Cross-Validation RMSE of 1.4189, Test RMSE of 1.4154, and an R^2 score of 0.9596, the Gradient Boosting model outperformed the others, indicating its robust predictive accuracy and effectiveness. Despite minor differences in MAE, Gradient Boosting consistently showed the best results, making it the optimal choice for our predictive tasks. This success underscores the model's capacity to generalize well to new, unseen data, affirming its suitability for deployment.

**new data testing step 1 for testing new data :Testing code with the totally new data of players 2022 that the machine did not see at all.**
"""

#Testing code with the totally new data of players 2022 that the machine did not see at all.
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
#from scipy.sparse import csr_matrix
import joblib

# Mount Google Drive if using Google Colab
from google.colab import drive
drive.mount('/content/drive')





# Define file paths
test_file_path = '/content/drive/My Drive/players_22 (1).csv'
model_file_path = '/content/drive/My Drive/Colab Notebooks/GradientBoosting.pkl'

# Load and display the data
print("Sample of the loaded data:")
test_data = pd.read_csv(test_file_path)
print(test_data.head())

"""The preprocess_and_predict function automates the preprocessing and prediction of new data using a pre-trained machine learning model. It begins by separating numerical and categorical data from the input test_data, ensuring each subset is handled appropriately. Missing values in numerical data are imputed using either the median or mean strategy based on the distribution of each feature. Categorical data missing values are filled with the mode of each column. Categorical variables are then one-hot encoded to convert them into a numerical format suitable for modeling.

Function to preprocess and predict new data because we shall have efficiency in doing that
"""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Mount Google Drive if using Google Colab
from google.colab import drive
drive.mount('/content/drive')

# Define file paths
test_file_path = '/content/drive/My Drive/players_22 (1).csv'
model_file_path = '/content/drive/My Drive/Colab Notebooks/GradientBoosting.pkl'
scaler_path = '/content/drive/My Drive/scaler.pkl'  # Path to the saved scaler

# Load and display the data
print("Sample of the loaded data:")
test_data = pd.read_csv(test_file_path)
print(test_data.head())

# Function to preprocess and predict new data, and calculate performance metrics
def preprocess_and_predict(test_data, model_file_path, scaler_path):
    # Separate numerical and categorical data
    numerical_cols = ['potential', 'age', 'shooting', 'passing', 'physic', 'movement_reactions']
    categorical_cols = ['preferred_foot']  # Add more if there are additional categorical columns
    target_col = 'overall'  # Target column

    # Ensure a copy to avoid SettingWithCopyWarning
    numerical_data = test_data[numerical_cols].copy()
    categorical_data = test_data[categorical_cols].copy()

    # Check if the target column exists in the test data
    if target_col not in test_data.columns:
        raise ValueError(f"Target column '{target_col}' not found in the test data.")

    true_values = test_data[target_col]

    # Impute missing values for numerical data
    for column in numerical_data.columns:
        imputer = SimpleImputer(strategy='median') if numerical_data[column].median() < numerical_data[column].mean() else SimpleImputer(strategy='mean')
        numerical_data.loc[:, column] = imputer.fit_transform(numerical_data[[column]])

    # Handle missing values for categorical data
    for column in categorical_data.columns:
        categorical_data.loc[:, column].fillna(categorical_data[column].mode()[0], inplace=True)

    # One-hot encode categorical data
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    categorical_data_encoded = encoder.fit_transform(categorical_data)

    # Get feature names after one-hot encoding
    categorical_feature_names = encoder.categories_[0].tolist()
    categorical_data_encoded = pd.DataFrame(categorical_data_encoded, columns=categorical_feature_names)

    # Combine numerical and encoded categorical data
    combined_data = pd.concat([numerical_data, categorical_data_encoded], axis=1)

    # Remove highly correlated features (if any)
    def correlation(X_numerical, threshold):
        col_corr = set()
        corr_matrix = pd.DataFrame(X_numerical).corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    colname = corr_matrix.columns[i]
                    col_corr.add(colname)
        return col_corr

    cor_features = correlation(combined_data, 0.7)
    combined_data.drop(columns=cor_features, inplace=True, errors='ignore')

    # Load the scaler from Google Drive
    scaler = joblib.load(scaler_path)

    # Standardize features using the loaded scaler
    combined_data_sparse = combined_data.astype(pd.SparseDtype("float", 0))
    X_scaled = scaler.transform(combined_data_sparse)

    # Load the model
    model_loaded = joblib.load(model_file_path)

    # Make predictions
    predictions = model_loaded.predict(X_scaled)

    # Calculate performance metrics
    mae = mean_absolute_error(true_values, predictions)
    mse = mean_squared_error(true_values, predictions)
    rmse = mean_squared_error(true_values, predictions, squared=False)
    r2 = r2_score(true_values, predictions)

    return predictions, true_values, mae, mse, rmse, r2

# Call the function and get predictions and performance metrics
predictions, true_values, mae, mse, rmse, r2 = preprocess_and_predict(test_data, model_file_path, scaler_path)

# Display the predictions and performance metrics
print("Predictions for new data:")
print(predictions)
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²): {r2}")

# Visual Evaluation: Plotting the Predicted vs Actual Values
plt.figure(figsize=(10, 6))
plt.scatter(true_values, predictions, alpha=0.7)
plt.plot([true_values.min(), true_values.max()], [true_values.min(), true_values.max()], 'k--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values')
plt.grid(True)
plt.show()

# Plotting the distribution of errors
errors = true_values - predictions
plt.figure(figsize=(10, 6))
sns.histplot(errors, bins=30, kde=True)
plt.xlabel('Prediction Error')
plt.title('Distribution of Prediction Errors')
plt.grid(True)
plt.show()



"""Conclusion:
Despite the performance drop on the new data, the model still performs reasonably well, with an R² value above 0.9. This suggests that the Gradient Boosting model remains robust and is still the best model for the given task. The decrease in performance metrics can be attributed to natural variations and differences between the training and new data sets.
"""