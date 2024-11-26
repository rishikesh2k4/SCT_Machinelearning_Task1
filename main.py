import pandas as pd

train_data = pd.read_csv('/content/train.csv')
test_data = pd.read_csv('/content/test.csv')

print(train_data.info())

print(test_data.describe())

features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
X = train_data[features]
y = train_data['SalePrice']

X = X.fillna(0)
test_data = test_data[features].fillna(0)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error
predictions = model.predict(X_val)
mse = mean_squared_error(y_val, predictions)
print(f'RMSE: {mse ** 0.5}')

test_predictions = model.predict(test_data)

submission = pd.DataFrame({'SalePrice': test_predictions})
submission.index.name = 'Id'
submission.to_csv('submission.csv', index=True)


# Function to predict price based on user input
def predict_house_price(gr_liv_area, bedroom_abv_gr, full_bath):
    # Create a DataFrame with the input features
    input_data = pd.DataFrame({
        'GrLivArea': [gr_liv_area],
        'BedroomAbvGr': [bedroom_abv_gr],
        'FullBath': [full_bath]
    })
    
    # Preprocess the input data (fill missing values)
    input_data = input_data.fillna(0)
    
    # Predict the house price using the trained model
    predicted_price = model.predict(input_data)
    
    return predicted_price[0]

# Get user input for the features
gr_liv_area = float(input("Enter Ground Living Area (in sqft): "))
bedroom_abv_gr = int(input("Enter number of bedrooms above ground: "))
full_bath = int(input("Enter number of full bathrooms: "))

# Call the prediction function with the input values
predicted_price = predict_house_price(gr_liv_area, bedroom_abv_gr, full_bath)

# Output the predicted price
print(f"The predicted house price is: ${predicted_price:.2f}")
