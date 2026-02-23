# 🚗 Car Price Predictor

[![Python Version](https://img.shields.io/badge/python-3.11.9-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![ML Framework](https://img.shields.io/badge/ML-PyCaret-orange.svg)](https://pycaret.org/)

## 📌 About the Project
[**Car Price Predictor**](https://car-price-predictor-y0am.onrender.com) is a full-stack machine learning web application that predicts the market value of used cars based on their specific features (brand, model, manufacturing year, mileage, engine capacity, power, etc.). 

The project features a robust Machine Learning pipeline built with PyCaret and a fast, responsive backend API developed with FastAPI, serving a user-friendly HTML/JS frontend.

## 📸 UI Showcase
<img width="2880" height="1800" alt="image" src="https://github.com/user-attachments/assets/5e176437-2ba2-437a-8ee7-060f5ea6613d" />
<img width="2880" height="1800" alt="image" src="https://github.com/user-attachments/assets/1c56fc76-0533-4d1f-a27e-4f44af380f6d" />

## 📚 API Documentation

The backend is built with FastAPI, which automatically generates comprehensive and interactive API documentation. You can explore the endpoints, request schemas, and test the API directly through the browser.

### 1. Swagger UI
An interactive interface that allows you to test API endpoints directly (using the "Try it out" feature).
* **Live Documentation:** [Swagger UI Docs](https://car-price-predictor-y0am.onrender.com/docs) *(For local run: `http://127.0.0.1:8000/docs`)*

<img width="2880" height="1800" alt="image" src="https://github.com/user-attachments/assets/bee6268e-a25a-47ca-a0a2-dcd3e1377d35" />

### 2. ReDoc
A clean, three-panel static documentation layout, perfect for reading and understanding the API structure and Pydantic models.
* **Live Documentation:** [ReDoc Docs](https://car-price-predictor-y0am.onrender.com/redoc) *(For local run: `http://127.0.0.1:8000/redoc`)*

<img width="2880" height="1800" alt="image" src="https://github.com/user-attachments/assets/69383237-be14-4ffc-8625-7eed7ba1ed89" />

## 🧠 Machine Learning Pipeline
The complete process of exploratory data analysis, feature engineering, and model tuning is documented in the `model_creation/model_creation.ipynb` notebook.

* **Dataset:** [Vehicle dataset](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho?select=car+details+v4.csv) (originally over 2,000 records).
* **Feature Engineering & Preprocessing:**
  * Currency conversion from INR to USD.
  * Extracted numeric values from string columns using Regex (e.g., Engine cc, Max Power hp, Max Torque Nm).
  * Converted Brake Horsepower (bhp) to metric Horsepower (hp).
  * Applied Target Encoding for high-cardinality categorical features (`Brand`, `Model`).
  * Used Simple Imputation for handling missing values.
* **Algorithm:** **K-Nearest Neighbors (KNN) Regressor** * Tuned over 50 iterations optimizing for R² using PyCaret.
* **Model Performance:**
  * **R² Score:** 0.839 (The model explains ~84% of the variance in car prices)
  * **MAE (Mean Absolute Error):** $2,268.85
  * **RMSE (Root Mean Squared Error):** $11,929.57

## 🛠️ Tech Stack
* **Backend Framework:** FastAPI, Uvicorn, Pydantic (for data validation)
* **Machine Learning:** PyCaret, Pandas, NumPy, scikit-learn
* **Frontend:** HTML, CSS, JavaScript (Jinja2 Templates)
* **Deployment/Model Serialization:** Pickle (`deployment.pkl`)

