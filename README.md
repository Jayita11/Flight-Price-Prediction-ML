# ✈️ Flight Price Prediction

## 📄 Overview

This project aims to develop a predictive model for flight prices using various features such as search date, departure date, airline, number of stops, and more. The goal is to provide insights into the factors affecting flight prices and build a robust model to predict future prices accurately.

![DALL·E 2024-08-16 22 31 58 - A second flat illustration for predicting flight prices with no text  The design should feature elements like a simplified airplane, a calendar, a gra](https://github.com/user-attachments/assets/25332ada-dddd-477d-b660-70b3b2563c12)

## 📑 Table of Contents

1. [📚 Libraries and Dataset](#libraries-and-dataset)
2. [⚙️ Data Preprocessing](#data-preprocessing)
3. [📊 Exploratory Data Analysis](#exploratory-data-analysis)
4. [🏋️‍♂️ Model Training & Selection](#model-training-&-Selection)
5. [📈 Model Evaluation](#model-evaluation)
6. [🚀 Deployment](#deployment)
7. [💻 Run Instructions](#run-instructions)
8. [📝 Conclusion](#conclusion)

## 📚 Libraries and Dataset

The project utilizes the following key libraries:
- **Pandas** 🐼: For data manipulation and analysis.
- **NumPy** 📐: For numerical computations.
- **Matplotlib & Seaborn** 📊: For data visualization.
- **Scikit-learn** 🤖: For model building, evaluation, and deployment.

The dataset includes information on flight searches and their associated prices. Key features include:
- **Searched Date**: Date when the flight search was conducted.
- **Departure Date**: Date of flight departure.
- **Arrival Date**: Expected date of arrival.
- **Flight Lands Next Day**: Indicator for whether the flight lands the next day.
- **Departure Airport**: Airport code or name of departure.
- **Arrival Airport**: Airport code or name of arrival.
- **Number Of Stops**: Number of stops between departure and arrival.
- **Route**: Path or sequence of stops from departure to arrival.
- **Airline**: The operating airline.
- **Cabin**: Cabin class (e.g., Economy, Business).
- **Price**: Flight ticket price.

## ⚙️ Data Preprocessing

Data preprocessing steps included:
- **Handling Missing Values** 🧩: Imputation of missing values to ensure data completeness.
- **Feature Engineering** 🔧: Creating new features that might be more predictive.
- **Encoding Categorical Variables** 🔢: Converting categorical variables into numerical format.
- **Splitting the Data** ✂️: Dividing the data into training and test sets for model evaluation.

## 📊 Exploratory Data Analysis (EDA)

EDA was performed to understand the relationships and distributions of various features. Key insights were derived by visualizing:
- **Price Distribution** 📈: Understanding how flight prices vary across different airlines, cabin classes, and other factors.
- **Correlation Analysis** 🔗: Identifying relationships between the features and the target variable (Price).

## 🏋️‍♂️ Model Training & Selection

The model training process began with a **Base Model** using Ordinary Least Squares (OLS) Linear Regression to establish a performance benchmark. This model, while simple, provided an initial understanding of how the features relate to the target variable (Price).

### 🧠 Additional Models

To improve upon the baseline, several other regression models were tested:
- **Ridge Regressor** 🌉: A variant of linear regression that includes a regularization term to prevent overfitting by penalizing large coefficients.
- **Lasso Regressor** 🧬: Another regularized linear model that can lead to sparse solutions, effectively selecting important features by driving the coefficients of less important features to zero.
- **Gradient Boosting Regressor** 🌱: A powerful ensemble method that builds multiple weak learners (typically decision trees) sequentially, where each new model attempts to correct errors made by the previous ones.
- **XGBoost Regressor** 🚀: An optimized implementation of gradient boosting, known for its high performance and speed, often used in competitive machine learning tasks.
- **Random Forest Regressor** 🌳: An ensemble of decision trees, where each tree is trained on a random subset of data and features, and the final prediction is an average of all the trees. This model is particularly robust to overfitting and can capture complex interactions between features.

### 🥇 Best Model Selection

After evaluating all the models, the **Random Forest Regressor** was selected as the best model. This decision was based on its superior performance in terms of accuracy and its ability to handle a wide variety of data distributions and feature interactions. The Random Forest model outperformed others in key metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-Squared (R²).

This model's ability to generalize well to unseen data, while maintaining stability and robustness, made it the ideal choice for predicting flight prices in this project.

## 📈 Model Evaluation

The models were evaluated using metrics such as:
- **Mean Absolute Error (MAE)** 📏
- **Root Mean Squared Error (RMSE)** 🧮
- **R-Squared (R²)** 🔢

The best-performing model **Random Forest Regressor** was selected based on these metrics.

**Values for Base Model**: 

![Screenshot 2024-08-16 at 10 25 15 PM](https://github.com/user-attachments/assets/09665d61-7533-4f4a-a3b0-61e9c605347c)

**Values for Best Model**:

![Screenshot 2024-08-16 at 10 17 09 PM](https://github.com/user-attachments/assets/bfb6ba67-2e54-4de2-9bbb-5d027c149a2c)

### 🔍 Insights

- Non-linear models, such as Random Forest and XGBoost, significantly outperformed linear models like OLS, Ridge, and Lasso Regression in predicting flight prices.
- **Random Forest Regressor** emerged as the best model with an R² of 0.7460, capturing complex interactions between features.
- **XGBoost Regressor** also performed strongly, benefiting from its ability to handle missing data and regularization techniques that prevent overfitting.

### 🧩 Feature Importance

An important aspect of the Random Forest model is its ability to provide insights into the importance of different features in making predictions. A feature importance plot was generated to visualize which features had the most significant impact on predicting flight price.

![Screenshot 2024-08-16 at 10 17 18 PM](https://github.com/user-attachments/assets/2950e63b-bf09-41d4-a972-6fd5efacb5ab)

### 🔍 Insights
- **Travel Time** was the most influential feature, followed by **Airline** and **Number of Stops**.
- Less important features included **Cabin** and **Flight Lands Next Day**, indicating they have a minimal impact on price prediction.

## 🚀 Deployment

Deploy the model using a Streamlit app (`app.py`). The app allows users to input flight price prediction data and get price predictions. To use the app, follow the link provided below:

[🔗 Streamlit App](#)

## 💻 Run Instructions

If you wish to run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/Jayita11/Flight-Price-Prediction-ML]
   cd Flight-Price-Prediction-ML

   python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`

pip install -r requirements.txt

streamlit run app.py

Access the app:
Open your browser and go to http://localhost:8501 to use the flight price prediction app.

## 📝 Conclusion

The project successfully demonstrated the ability to predict flight prices using machine learning techniques. The insights gained from the analysis can be valuable for airlines and consumers alike in understanding the dynamics of flight pricing.
