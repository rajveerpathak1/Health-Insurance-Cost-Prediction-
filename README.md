# Health Insurance Cost Prediction (Feb 2025)

This project focuses on predicting health insurance costs based on demographic data using machine learning models. The project includes hypothesis testing to identify significant factors influencing health insurance premiums.

## Features
- **Machine Learning Model Development**: Built multiple regression models to predict insurance costs based on features such as age, BMI, smoking habits, etc.
- **Hypothesis Testing**: Applied statistical techniques to identify key cost-influencing factors.
- **Model Evaluation**: Used metrics like MAE (Mean Absolute Error) and MSE (Mean Squared Error) to evaluate model performance.
# Health Insurance Cost Prediction


## 📂 Dataset
The dataset used for training is **medical_insurance.csv**, which contains the following features:
- `age`: Age of the individual
- `sex`: Gender (male/female)
- `bmi`: Body Mass Index (BMI)
- `children`: Number of children covered by health insurance
- `smoker`: Whether the individual is a smoker (yes/no)
- `region`: Residential area in the US (northwest, southwest, northeast, southeast)
- `charges`: Medical insurance cost (target variable)

##  Tech Stack
- **Programming Language**: Python
- **Libraries Used**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Joblib
- **Model Used**: Regression models (Linear Regression, Random Forest, etc.)

##  Project Structure
```
Health_Cost_Prediction/
│-- medical_insurance.csv
│-- Health_Cost_Prediction.ipynb
│-- insurance_cost_model.pkl (Trained Model - Joblib)
│-- requirements.txt
│-- README.md
```

##  Installation & Execution
###  Clone the Repository
```bash
git clone https://github.com/rajveerpathak1/Health-Insurance-Cost-Prediction.git
cd Health-Insurance-Cost-Prediction
```

###  Install Dependencies
```bash
pip install -r requirements.txt
```

###  Run the Jupyter Notebook
```bash
jupyter notebook
```
Open **Health_Cost_Prediction.ipynb** and execute the cells to train the model.

###  Load and Use the Trained Model
To use the pre-trained model:
```python
import joblib
model = joblib.load("insurance_cost_model.pkl")
prediction = model.predict([[age, bmi, children, smoker, region]])
print("Predicted Insurance Cost:", prediction)
```

##  Results
- Achieved Good level of accuracy in predicting insurance costs.
- Identified that **smoking** and **BMI** have a significant impact on insurance charges.

##  Contribution
Feel free to fork the repo, raise issues, or contribute via pull requests!


