import streamlit as st
import pandas as pd
import joblib

# Set the title with an emoji
st.title("✈️ Flight Price Prediction")

# Load the dataset to get airport names
df_flight = pd.read_csv('/Users/jayitachatterjee/Desktop/ML_Projects/Flight_Price_Prediction/Dataset/flights.csv')
unique_departure_airports = df_flight['Departure Airport'].unique()
unique_arrival_airports = df_flight['Arrival Airport'].unique()

# Load the model and scaler
model_dict = joblib.load('artifacts/model_data.joblib')
model = model_dict['model']
scaler = model_dict['scaler']

# Mappings for categorical variables
airline_mapping = {
    'Alaska Airlines': 1,
    'Delta': 2,
    'American Airlines': 3,
    'Spirit Airlines': 4,
    'United Airlines': 5
}

airport_mapping = {
    'LAX': 1,
    'SFO': 2,
    'BOS': 3,
    'JFK': 4,
    'ORD': 5,
    'LAS': 6
}

cabin_mapping = {
    'Economy': 1,
    'Business': 2,
    'First': 3
}

# Input fields for the user to enter flight details
st.header("Enter Flight Details")

flight_lands_next_day = st.selectbox('Does the flight land the next day? 🌙', ['Yes', 'No'])

# Map the user's selection to the corresponding numerical value
flight_lands_next_day = 1 if flight_lands_next_day == 'Yes' else 0

departure_airport = st.selectbox('Departure Airport 🛫', airport_mapping.keys())
arrival_airport = st.selectbox('Arrival Airport 🛬', airport_mapping.keys())
number_of_stops = st.number_input('Number of Stops ⛔️', min_value=0, step=1)
airline = st.selectbox('Airline 🛩️', airline_mapping.keys())
cabin = st.selectbox('Cabin Class 🎟️', cabin_mapping.keys())
days_before_travel = st.number_input('Days Before Travel 📅', min_value=0, step=1)
travel_time = st.number_input('Travel Time (in hours) ⏰', min_value=0.0, step=0.1)

# Predict buttonif st.button('Predict 🚀'):
    # Map the inputs using the provided dictionaries
input_data = pd.DataFrame({
        'Flight Lands Next Day': [flight_lands_next_day],
        'Departure Airport': [airport_mapping[departure_airport]],
        'Arrival Airport': [airport_mapping[arrival_airport]],
        'Number Of Stops': [number_of_stops],
        'Airline': [airline_mapping[airline]],
        'Cabin': [cabin_mapping[cabin]],
        'DaysBeforeTravel': [days_before_travel],
        'TravelTime': [travel_time]
    })

    # Scale the inputs using the loaded scaler
input_data_scaled = scaler.transform(input_data)

    # Predict the price using the modeltry:
prediction = model.predict(input_data_scaled)
if st.button('Predict'):
     st.write(f"Predicted Flight Price: ${prediction[0]:.2f}")