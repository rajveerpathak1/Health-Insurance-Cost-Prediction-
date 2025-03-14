# Health Insurance Cost Prediction (Feb 2025)

This project focuses on predicting health insurance costs based on demographic data using machine learning models. The project includes hypothesis testing to identify significant factors influencing health insurance premiums.

## Features
- **Machine Learning Model Development**: Built multiple regression models to predict insurance costs based on features such as age, BMI, smoking habits, etc.
- **Hypothesis Testing**: Applied statistical techniques to identify key cost-influencing factors.
- **Model Evaluation**: Used metrics like MAE (Mean Absolute Error) and MSE (Mean Squared Error) to evaluate model performance.

## 📂 Dataset
The dataset used for training is **medical_insurance.csv**, which contains the following features:
- `age`: Age of the individual
- `sex`: Gender (male/female)
- `bmi`: Body Mass Index (BMI)
- `children`: Number of children covered by health insurance
- `smoker`: Whether the individual is a smoker (yes/no)
- `region`: Residential area in the US (northwest, southwest, northeast, southeast)
- `charges`: Medical insurance cost (target variable)

## Tech Stack
- **Programming Language**: Python
- **Libraries Used**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Joblib, Flask, Gunicorn
- **Model Used**: Regression models (Linear Regression, Random Forest, etc.)
- **Deployment**: Render (Flask API with Gunicorn)

## Project Structure
```
Health_Cost_Prediction/
│-- medical_insurance.csv
│-- Health_Cost_Prediction.ipynb
│-- insurance_cost_model.pkl (Trained Model - Joblib)
│-- app.py (Flask API for deployment)
│-- requirements.txt
│-- README.md
```

## Installation & Execution
### Clone the Repository
```bash
git clone https://github.com/rajveerpathak1/Health-Insurance-Cost-Prediction.git
cd Health-Insurance-Cost-Prediction
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Jupyter Notebook
```bash
jupyter notebook
```
Open **Health_Cost_Prediction.ipynb** and execute the cells to train the model.

### Load and Use the Trained Model
To use the pre-trained model:
```python
import joblib
model = joblib.load("insurance_cost_model.pkl")
prediction = model.predict([[age, bmi, children, smoker, region]])
print("Predicted Insurance Cost:", prediction)
```

## Deployment on Render
### Steps to Deploy
1. **Ensure all dependencies are in `requirements.txt`**.
2. **Push your project to GitHub**.
3. **Create a new Render Web Service**:
   - Select your GitHub repo.
   - Use **`gunicorn app:app`** as the start command.
4. **Upload `insurance_cost_model.pkl` to the Render project directory**.
5. **Update `app.py` to load the model using a relative path**:
```python
import joblib
import os

model_path = os.path.join(os.path.dirname(__file__), "insurance_cost_model.pkl")
model = joblib.load(model_path)
```
6. **Deploy and test your API** at `https://your-app.onrender.com`.
7. ** currently deployed** at 'https://health-insurance-cost-prediction-3.onrender.com' .

## Results
- Achieved a good level of accuracy in predicting insurance costs.
- Identified that **smoking** and **BMI** have a significant impact on insurance charges.

## Contribution
Feel free to fork the repo, raise issues, or contribute via pull requests!
