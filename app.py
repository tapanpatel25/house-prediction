# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from flask import Flask, request, render_template

# Initialize Flask app
app = Flask(__name__)

# Load the data from the CSV file (replace 'your_data.csv' with the actual filename)
df = pd.read_csv('house_data.csv')

# Separate features (X) and target (y)
X = df[['num_bedrooms', 'num_bathrooms', 'square_footage', 'lot_size', 'age_of_house', 'proximity_to_city_center', 'neighborhood_quality']]
y = df['house_price']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Define a route for the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get input values from the form
        num_bedrooms = float(request.form['num_bedrooms'])
        num_bathrooms = float(request.form['num_bathrooms'])
        square_footage = float(request.form['square_footage'])
        lot_size = float(request.form['lot_size'])
        age_of_house = float(request.form['age_of_house'])
        proximity_to_city_center = float(request.form['proximity_to_city_center'])
        neighborhood_quality = float(request.form['neighborhood_quality'])
        
        # Create a DataFrame for the input values
        input_data = pd.DataFrame({
            'num_bedrooms': [num_bedrooms],
            'num_bathrooms': [num_bathrooms],
            'square_footage': [square_footage],
            'lot_size': [lot_size],
            'age_of_house': [age_of_house],
            'proximity_to_city_center': [proximity_to_city_center],
            'neighborhood_quality': [neighborhood_quality]
        })
        
        # Predict the house price
        predicted_price = model.predict(input_data)[0]
        result = f"${predicted_price:,.2f}"
    else:
        result = None

    return render_template('index.html', result=result)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
