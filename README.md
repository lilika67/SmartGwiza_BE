#  SmartGwiza - Maize Yield Prediction System for Rwanda

##  Video Demo

To see the SmartGwiza  demo you can  open[Video demo](https://www.loom.com/share/cf035345c65a41ee99a3c00db452fb1f)

## Mission

SmartGwiza is an intelligent agricultural platform that uses machine learning to predict maize crop yields in Rwanda. Our system helps farmers , agricultural experts, and policymakers make data-driven decisions by analyzing environmental factors, soil conditions, and farming practices to provide accurate yield predictions.


## Overview
This repository contains a machine learning project to predict maize yield in Rwanda using historical data .  We used two different dataset where one that was used for national-level prediction was got from food and agriculture organisation and another was generated with realistic data from NISR(National institute of statistics of Rwanda) different surveys.

Core Features

**AI-Powered Yield Prediction** Get accurate maize yield predictions based on multiple parameters

**Real-time Validation** : Input validation with optimal range indicators

**Prediction History** : Track and analyze past predictions

**Yield Data Submission** : Contribute to model improvement with actual harvest data

**User and Admin Dashboard**


SmartGwiza is built with a **three-part architecture**:
1. **Frontend** – User interface for interaction.
2. **Backend (FastAPI)** – API services and endpoints.
3. **Machine Learning component** – Predictive modeling and data analysis.

The National level datasets includes features such as:
- `Year`
- `pesticides_tonnes`
- `avg_temp`

The farmer-level dataset includes features such as:

- `District`
- `Pesticides used(kg/ha)`
- `Fertilizer(l/ha)`
- `Irrigation method`
- `Rainfall(mm)`
- `Temperature(c)`

These features support agricultural planning and policy development for farmers and also other people since many are great factors that influence the productivity of maize.

### Implemented Models
1. **Custom Neural Network (`MaizeYieldNN`)**
2. **Linear Regression**
3. **Polynomial Regression (Degree 2)**

>  The Neural Network achieved the best performance:
- R²: **0.931**
- MSE: **335,306**
- MAE: **315**

The model was saved as **`maize_yield_model.pth`** to be used in making predictions.

##  Dataset
The dataset is a CSV file with **33 rows**, containing:

| Feature | Description | Range |
|----------|--------------|--------|
| `Year` | Year of record | 1990–2023 (gap in 2003) |
| `hg/ha_yield` | Maize yield (hg/ha) | 10,252–22,845 |
| `pesticides_tonnes` | Pesticide use (tonnes) | 97–2,500 |
| `avg_temp` | Average temperature (°C) | 19.22–20.29 |

Farmer-level dataset


| Feature | Description | Range |
|----------|--------------|--------|
| `country` | Country for agricultural activity | Rwanda |
| `District` | Distict | Rwandan district |
| `pesticides_tonnes` | Pesticide use (tonnes) | 7-12(l/ha) |
| `temperature` | Average temperature (°C) | 18-21 |
| `Fertilizer` | amount of npk fertilizer used | 60-80kg/ha |
| `Irrigation type` | Method of irrigation they will be using| Rainfed/Irrigated |
| `Rainfall` | Rainfall | 500-1400mm |
| `soil ph` | measure of soil acidity | 7-12l/ha |


##  Model performance Details

For the farmer-level data 

<img width="1074" height="664" alt="image" src="https://github.com/user-attachments/assets/15d78817-b47d-420d-8e0b-53bdf626a467" />


Best model saved as rwanda_maize_yield_predictor.pkl

For Nationa-level data

Model Comparison:
Neural Network: R²=0.945, MSE=265886, MAE=272
Linear Regression: R²=0.931, MSE=338282, MAE=395
Polynomial Regression: R²=0.705, MSE=1434811, MAE=967

Best Model: Neural Network (saved as 'maize_yield_model.pth') with R²=0.945, MSE=265886, MAE=272.

##  Saved Model
The best performing model (`maize_yield_model.pth`) is saved and automatically loaded during predictions.  
Ensure this file is available in your project directory before running the API.

 
## Features
- **Data Analysis**  
  Exploratory Data Analysis (EDA) using visualizations such as histograms, scatter plots, and correlation matrices.
  
- **Model Training**  
  Implements and compares three predictive models using PyTorch and NumPy.

- **Prediction Tool**  
  Includes a `predict_yield` function with input validation and range checking.

## How to run backend and Dependency Installation

### 1. Clone the Repository
```bash
git clone https://github.com/lilika67/SmartGwiza_System.git
cd SmartGwizaS
```

### 2. Install Dependencies
Ensure Python **3.8+** and `pip` are installed, then run:
```bash
pip install -r requirements.txt
```

Your `requirements.txt` should include:
```
setuptools>=69.0.0
wheel
numpy<2.0
scikit-learn==1.6.1
torch
torchvision  
torchaudio
fastapi
uvicorn[standard]
pydantic
requests
```

### 3. Set Up a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```
Add .env file that contain 
```bash
MONGODB_URL 
SECRET_KEY
ALGORITHM

```

## 2. How to run backend and Dependency Installation

## Installation

1. Clone the repository:

```sh
git clone (https://github.com/lilika67/smartgwiz-system.git)
cd smartgwiz-system
```


2. Install dependencies:

```sh
npm install
```


3. Start the development server:

```sh
npm run dev
```


4. Open [http://localhost:3000](http://localhost:3000) in your browser to see the application.


##  Notebooks usage

### Run the Jupyter Notebook
Open SmartGwiza_System navigate to Notebook folder and run the notebooks in Jupyter Notebook or Google Colab and run all cells to:
- Load and explore data  
- Train models  
- Evaluate performance  

## Frontend platform usage

**User Registration & Authentication**

**Sign Up** : Create a new farmer account

**Login** : Access your personalized dashboard

**Profile Management** : Update personal information

**Yield Prediction**

**Navigate to Predict Tab** : Access the prediction form

**Input Farming Parameters**:

**Get AI Prediction** : Receive instant yield forecast with category levels

 **Data Submission**
**Submit Actual Yield** : Help improve the model by sharing real harvest data

**View History** : Track your prediction accuracy over time

**Compare Results** : Analyze predicted vs actual yields


##  FastAPI Integration

### Run Locally
To start the FastAPI backend, run:
```bash
uvicorn main:app --reload
```

Then open the interactive **Swagger UI** at:  

http://localhost:8000/docs


###  API Endpoints

#### **GET /predict**
Predict maize yield based on input parameters.
 


## Deployment

- **Machine Learning:** Developed and trained using Jupyter Notebook.  
- **Backend:** Implemented with FastAPI and deployed on **Render**.  
- **Frontend:** User-friendly web app deployed on **Vercel**.


## Other Links

**Frontend Repository:**  
 [SmartGwiza Frontend](https://github.com/lilika67/SmartGwiza-system.git)

 **Frontend Link ([FrontendLink](https://smartgwiz-system.vercel.app/))**  
 
 **Backend  Link ([BackendLink](https://smartgwiza-be-1.onrender.com/docs#/))**  

 
## Summary

SmartGwiza integrates **data science**, **AI**, and **web technologies** to provide actionable insights for Rwanda’s agriculture.  
By combining historical trends and predictive models, it empowers decision makers to plan effectively for the future of maize production.

## Conclusion and Recommendation:

**Conclusion**

This project of SmartGwiza successfully demonstrates how machine learning can be applied to predict maize yield in Rwanda using available agricultural and environmental data. Through systematic data cleaning, visualization, and model development, the notebook highlights the most influential factors affecting yield and provides a reproducible pipeline for predictive analytics.

The final model delivers valuable insights that can guide decisions on crop management and planning. Future improvements may include incorporating additional datasets such as soil quality, rainfall distribution, or satellite imagery, as well as deploying the model into a real-time decision support tool for farmers and agricultural institutions.

**Practical recommendations**

To get the most from SmartGwiza, regularly input accurate field data and use the real-time validation feedback to optimize your farming practices. Submit your actual yields after harvest to help improve the AI model, and analyze your prediction history to identify successful patterns. Use the prediction history to track your progress and learn from seasonal patterns and get insights to refine your farming strategies season after season.

## Related screenshot of UI

1. Home page
   
   <img width="2880" height="1456" alt="image" src="https://github.com/user-attachments/assets/915c9811-b08c-4027-8946-a7adbabbf13d" />

2.How it works page

<img width="2876" height="1466" alt="image" src="https://github.com/user-attachments/assets/b34cddc4-ee30-4ed7-9f4e-533465786b4f" />

3. Visualization page

   <img width="2866" height="1478" alt="image" src="https://github.com/user-attachments/assets/c3f96e13-ca9a-4830-9370-6f7b7279054b" />

4. Login and Signup page

   <img width="2846" height="1484" alt="image" src="https://github.com/user-attachments/assets/805663e3-285c-492e-aab1-80aaf138e97b" />

5. Yield prediction on national-level

   <img width="2874" height="1492" alt="image" src="https://github.com/user-attachments/assets/2d045198-e989-4a0a-a5ba-ed637373e409" />


6. UserDashboard
   Prediction form
   
<img width="2870" height="1462" alt="image" src="https://github.com/user-attachments/assets/20e653c5-ad63-402f-9371-b68a86fdf70d" />

   Feeedback form

   <img width="2810" height="1554" alt="image" src="https://github.com/user-attachments/assets/a95c57a7-8485-4ab7-872f-57725df3349b" />


 7.Admin Dashboard

 <img width="2870" height="1468" alt="image" src="https://github.com/user-attachments/assets/fe6d802d-966d-4678-a39a-9afe00acba80" />

 <img width="2854" height="1480" alt="image" src="https://github.com/user-attachments/assets/35a111e3-9422-4fae-9ba7-673093c66b87" />

 <img width="2880" height="1496" alt="image" src="https://github.com/user-attachments/assets/1aab3f9a-0aeb-4213-a85e-886723317047" />



   



