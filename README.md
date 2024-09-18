# Industrial Copper Price Modeling

![industrial copper modelling](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcREfRESMi8_ETc5m7RdkzirL-mHpjDxaY8AsA&s)


This project focuses on building predictive models for industrial copper prices. Using historical data and various economic and industrial factors, we develop models that aim to predict future copper prices. The project involves data preprocessing, feature engineering, model training, evaluation, and predictions to help in market analysis and decision-making.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Methodology](#methodology)
5. [Modeling](#modeling)
6. [Evaluation](#evaluation)
7. [Results](#results)
8. [Future Work](#future-work)
9. [Contributing](#contributing)
10. [License](#license)

---

## Project Overview

Copper is a critical industrial metal, used in sectors like electronics, construction, and energy. Price fluctuations in copper can significantly impact global industries. This project aims to build a predictive model that can forecast copper prices based on historical data, macroeconomic indicators, industrial production statistics, and market trends.

---

## Dataset

The dataset includes historical data of copper prices 

## Installation

### Requirements

To run the project, you will need the following installed:

- Python 3.8+
- Jupyter Notebook (optional)
- Libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`

### Steps

1. Clone the repository:
    ```bash
    git clone https://github.com/Sathyanagaraj03/industrial-copper-modeling.git
    cd industrial-copper-modeling
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Jupyter Notebook:
    ```bash
    jupyter notebook
    ```

---

## Methodology

1. **Data Collection**: Aggregate copper price data and relevant economic indicators.
2. **Data Preprocessing**:
   - Handle missing values
   - Feature scaling
   - Stationarity testing for time series analysis (ADF test)
3. **Feature Engineering**:
   - Lag features for time series modeling (e.g., copper price at t-1, t-7, etc.)
   - Rolling averages and moving windows (7-day, 30-day moving averages)
   - Incorporate external factors (e.g., crude oil prices, exchange rates, global economic growth)
4. **Modeling**:
   - Machine learning models such as Random Forest, Decision tree model for win or loss prediction.


---

## Modeling

We experiment with multiple models for copper price prediction:

1. **Decision Tree**: Traditional time series forecasting models.
2. **Random Forest**: A tree-based ensemble learning method.


## Evaluation

We evaluate model performance using various metrics, including:

- **Mean Absolute Error (MAE)**
- **Root Mean Square Error (RMSE)**
- **R² Score**
- **Mean Absolute Percentage Error (MAPE)**

We also plot prediction results and compare them to the actual historical copper prices to assess model accuracy visually.

---

## Results

- **Best Model**: The Decision Tree and Random Forest model 
- **Feature Importance**: Top contributing factors include:
  - Previous copper price (lagged values)
  - Crude oil price
  - Global industrial production index
  - Exchange rate (USD to major currencies)

---

## Future Work

- Incorporating more detailed industry-specific data, such as copper mining production and inventories.
- Implementing deep learning models like transformers for better time-series forecasting.
- Conducting real-time prediction by integrating APIs for real-time data feeds.
- Testing the model on other commodities such as aluminum or steel for cross-industry insights.

---

## Contributing

We welcome contributions! If you want to contribute, please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Create a Pull Request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

