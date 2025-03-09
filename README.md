# 🛒 Walmart Sales Prediction using LSTM  

## 📌 Overview  
This project implements a **Long Short-Term Memory (LSTM) model** to forecast **weekly sales for Walmart**. The model is trained on historical sales data, leveraging **time-series forecasting techniques** to predict future trends.  

By preprocessing the dataset, normalizing sales values, and using **sequential data** for training, the LSTM model learns patterns and provides accurate sales predictions. The project also includes **visualizations** to analyze sales trends and model performance.  

## 🚀 Features  
✅ **Data Preprocessing:** Handles missing values, normalizes sales data, and formats dates  
✅ **Sequential Data Preparation:** Converts sales data into **time-series sequences**  
✅ **LSTM Model Training:** Uses PyTorch-based LSTM with **Adam optimizer & MSE loss**  
✅ **Evaluation Metrics:** Computes **RMSE & MAE** for model performance  
✅ **Future Sales Forecasting:** Predicts upcoming sales trends  
✅ **Visualizations:** **Plots actual vs predicted sales, trends, and forecasts**  

## 📊 Dataset  
The dataset used is **Walmart Store Sales**, which contains:  

| Column          | Description                                |
|----------------|--------------------------------------------|
| `Store`        | Store ID                                   |
| `Date`         | Timestamp of weekly sales                 |
| `Weekly_Sales` | Total sales for the week                  |
| `Holiday_Flag` | Binary flag indicating a holiday week     |
| `Temperature`  | Temperature in the region                 |
| `Fuel_Price`   | Price of fuel in the region               |
| `CPI`          | Consumer Price Index                      |
| `Unemployment` | Unemployment rate in the region           |

## 🛠️ Tech Stack  
- **Python** – Core language  
- **Pandas & NumPy** – Data manipulation  
- **Matplotlib & Seaborn** – Data visualization  
- **Scikit-learn** – Data scaling & evaluation metrics  
- **PyTorch** – Deep learning (LSTM model)  

## 📌 Implementation Steps  
### **1️⃣ Load and Preprocess Data**  
- Read the dataset using **Pandas**  
- Convert the `Date` column to **datetime format** and set it as the index  
- Normalize `Weekly_Sales` using **MinMaxScaler**  
- Split data into **training (80%) and testing (20%) sets**  

### **2️⃣ Convert Data into Sequential Format**  
- Use a **custom PyTorch Dataset class** (`SalesDataset`)  
- Convert data into sequences of **10 time steps**  

### **3️⃣ Define LSTM Model**  
- Build an **LSTM model** with input, hidden, and output layers  
- Use **Adam optimizer** and **MSE loss function**  
- Train the model for multiple epochs with **batch gradient descent**  

### **4️⃣ Train the Model**  
- Implement a training loop with **forward propagation, backpropagation, and optimization**  
- Monitor **loss reduction over epochs**  

### **5️⃣ Evaluate Model Performance**  
- Compute **Root Mean Squared Error (RMSE)** and **Mean Absolute Error (MAE)**  
- Compare **predicted vs actual sales**  

### **6️⃣ Forecast Future Sales**  
- Use the trained LSTM model to **generate future sales predictions**  
- Convert predicted values back to original scale using **inverse transformation**  

## 🎯 Sample Results  
### **📊 Actual vs Predicted Sales**  
| Date       | Actual Sales | Predicted Sales |  
|------------|--------------|-----------------|  
| 2023-07-10 | 15,600       | 15,450          |  
| 2023-07-17 | 16,200       | 16,050          |  
| 2023-07-24 | 17,500       | 17,320          |  
| 2023-07-31 | 18,000       | 17,880          |  

## 📈 Visualizations  

### 1️⃣ **Sales Trends Over Time**  
- Analyzes Walmart’s historical **weekly sales**  
- Identifies patterns, seasonality, and anomalies  

### 2️⃣ **Actual vs Predicted Sales**  
- Compares model predictions with real sales data  
- Evaluates **model accuracy**    

### 3️⃣ **Future Sales Forecasting**  
- Predicts **upcoming sales** based on trained LSTM model  
- Provides insights for business planning  
"# Real-Time-Demand-Forecasting-System-for-Retail-" 
