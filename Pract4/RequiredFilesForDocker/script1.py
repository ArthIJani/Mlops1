import streamlit as st
import numpy as np
import pickle

# Load the pickled scaler and model
scaler = pickle.load(open("scaler", "rb"))
loaded_model = pickle.load(open("model", "rb"))

def value_predictor(to_predict_list):
    X_test = np.array(to_predict_list).reshape(1, 1)

    # Normalize the data
    X_test_normalized = scaler.transform(X_test)

    result = loaded_model.predict(X_test_normalized)
    return result[0]

def main():
    st.title("Value Predictor")

    st.write("Enter a value to predict:")

    # User input
    user_input = st.text_input("Enter a value:")

    if user_input:
        to_predict_list = [int(user_input)]
        prediction = value_predictor(to_predict_list)

        st.write("Prediction:", prediction)

if __name__ == "__main__":
    main()
