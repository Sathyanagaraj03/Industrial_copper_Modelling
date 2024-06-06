# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load data
def load_data():
    # Load your dataset here
    df = pd.read_csv('dataset.csv')  # Replace with the path to your dataset
    return df

# Function to train the Decision Tree model
def train_decision_tree(df):
    x = df[['qt_r', 'sp_r', 'application', 'thick_r', 'width', 'country', 'customer', 'product_ref']]
    y = df['status']

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return model, X_test, y_test, y_pred, accuracy, report, cm

# Function to train the Linear Regression model
def train_linear_regression(df):
    x = df[['qt_r','application', 'thick_r', 'width', 'country', 'customer', 'product_ref']]
    y = df['sp_r']

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, X_test, y_test, y_pred, mse, r2

# Main function
def main():
    st.title("Industrial Copper Modelling")
    df = load_data()
    options="Make Predictions"
    if options == "Make Predictions":
        st.header("Make Predictions")
        
        model_option = st.selectbox("Select Model", ["Decision Tree", "Linear Regression"])
    
        if model_option == "Decision Tree":
            model, X_test, y_test, y_pred, accuracy, report, cm = train_decision_tree(df)
            model, _, _, _, _, _, _ = train_decision_tree(df)
            
            qt_r = st.number_input("Enter value for qt_r")
            sp_r = st.number_input("Enter value for sp_r")
            application = st.text_input("Enter value for application")
            thick_r = st.number_input("Enter value for thick_r")
            width = st.number_input("Enter value for width")
            country = st.text_input("Enter value for country")
            customer = st.text_input("Enter value for customer")
            product_ref = st.text_input("Enter value for product_ref")
            
            if st.button("Predict"):
                input_data = np.array([[qt_r, sp_r, application, thick_r, width, country, customer, product_ref]])
                prediction = model.predict(input_data)
                st.write(f"Predicted status: {prediction[0]}")
        
        elif model_option == "Linear Regression":
            model, X_test, y_test, y_pred, mse, r2 = train_linear_regression(df)
            model, _, _, _, _, _ = train_linear_regression(df)
            
            qt_r = st.number_input("Enter value for qt_r")
            #sp_r = st.number_input("Enter value for sp_r")
            application = st.number_input("Enter value for application")
            thick_r = st.number_input("Enter value for thick_r")
            width = st.number_input("Enter value for width")
            country = st.number_input("Enter value for country")
            customer = st.number_input("Enter value for customer")
            product_ref = st.number_input("Enter value for product_ref")
            
            if st.button("Predict"):
                input_data = np.array([[qt_r, application, thick_r, width, country, customer, product_ref]])
                prediction = model.predict(input_data)
                st.write(f"Predicted selling price: {prediction[0]}")

if __name__ == "__main__":
    main()
