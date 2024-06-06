# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load data
def load_data():
    # Load your dataset here
    df = pd.read_csv('dataset.csv')  # Replace with the path to your dataset
    return df

# Function to train the model
def train_model(df):
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

# Main function
def main():
    st.title("Industrial Copper Modelling with Decision Tree Classifier")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Go to", ["Data Exploration", "Model Training", "Visualize Decision Tree", "Make Predictions"])
    
    df = load_data()
    
    if options == "Data Exploration":
        st.header("Data Exploration")
        st.write("Dataset Preview")
        st.write(df.head())
        
        st.write("Summary Statistics")
        st.write(df.describe())
        
    elif options == "Model Training":
        st.header("Model Training")
        
        model, X_test, y_test, y_pred, accuracy, report, cm = train_model(df)
        
        st.write(f"Model Accuracy: {accuracy}")
        st.write("Classification Report")
        st.text(report)
        
        st.write("Confusion Matrix")
        plt.figure(figsize=(10, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
        st.pyplot(plt)
    
    elif options == "Visualize Decision Tree":
        st.header("Visualize Decision Tree")
        
        model, _, _, _, _, _, _ = train_model(df)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        tree.plot_tree(model, filled=True, feature_names=['qt_r', 'sp_r', 'application', 'thick_r', 'width', 'country', 'customer', 'product_ref'], class_names=model.classes_)
        st.pyplot(fig)
    elif options == "Make Predictions":
        st.header("Make Predictions")
        
        model, _, _, _, _, _, _ = train_model(df)
        
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

if __name__ == "__main__":
    main()
