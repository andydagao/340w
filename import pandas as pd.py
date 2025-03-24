import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# Prepare the data
def prepare_data(df, product_name):
    # Melt the dataset to have columns: Year, Month, Price
    df_melted = df.melt(id_vars=["Year"], var_name="Month", value_name="Price")

    # Convert Month to a numeric value for regression
    month_map = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    df_melted['Month'] = df_melted['Month'].map(month_map)

    # Define the features and target variable
    X = df_melted[['Year', 'Month']]
    y = df_melted['Price']

    return X, y


# Function to train and evaluate a model
def train_and_evaluate_model(X, y, product_name):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error for {product_name}: {mse}')

    # Plotting the actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.title(f'Actual vs Predicted Prices for {product_name}')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.show()


# Prepare data for each product
X_beard, y_beard = prepare_data(beard_data, 'Beard per lb')
X_chicken, y_chicken = prepare_data(chicken_data, 'Chicken per lb')
X_cpi, y_cpi = prepare_data(cpi_data, 'CPI Food at Home')
X_eggs, y_eggs = prepare_data(eggs_data, 'Eggs per dozen')

# Train and evaluate models for each product
train_and_evaluate_model(X_beard, y_beard, 'Beard per lb')
train_and_evaluate_model(X_chicken, y_chicken, 'Chicken per lb')
train_and_evaluate_model(X_cpi, y_cpi, 'CPI Food at Home')
train_and_evaluate_model(X_eggs, y_eggs, 'Eggs per dozen')