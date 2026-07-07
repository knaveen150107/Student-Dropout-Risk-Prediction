from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# ==========================================================
# Load Saved Model
# ==========================================================

model = joblib.load("/home/naveen/Desktop/Student_Dropout/model/logistic_regression_model.pkl")
scaler = joblib.load("/home/naveen/Desktop/Student_Dropout/model/scaler.pkl")
label_encoder = joblib.load("/home/naveen/Desktop/Student_Dropout/model/label_encoder.pkl")

# ==========================================================
# FastAPI App
# ==========================================================

app = FastAPI(
    title="Student Dropout Prediction API",
    description="Predict whether a student will Dropout, Enrolled or Graduate",
    version="1.0"
)

# ==========================================================
# Input Schema
# ==========================================================

class StudentData(BaseModel):
    Marital_status: int
    Application_mode: int
    Application_order: int
    Course: int
    Daytime_evening_attendance: int
    Previous_qualification: int
    Previous_qualification_grade: float
    Nacionality: int
    Mothers_qualification: int
    Fathers_qualification: int
    Mothers_occupation: int
    Fathers_occupation: int
    Admission_grade: float
    Displaced: int
    Educational_special_needs: int
    Debtor: int
    Tuition_fees_up_to_date: int
    Gender: int
    Scholarship_holder: int
    Age_at_enrollment: int
    International: int
    Curricular_units_1st_sem_credited: int
    Curricular_units_1st_sem_enrolled: int
    Curricular_units_1st_sem_evaluations: int
    Curricular_units_1st_sem_approved: int
    Curricular_units_1st_sem_grade: float
    Curricular_units_1st_sem_without_evaluations: int
    Curricular_units_2nd_sem_credited: int
    Curricular_units_2nd_sem_enrolled: int
    Curricular_units_2nd_sem_evaluations: int
    Curricular_units_2nd_sem_approved: int
    Curricular_units_2nd_sem_grade: float
    Curricular_units_2nd_sem_without_evaluations: int
    Unemployment_rate: float
    Inflation_rate: float
    GDP: float


# ==========================================================
# Home Endpoint
# ==========================================================

@app.get("/")
def home():
    return {
        "message": "Student Dropout Prediction API is Running!"
    }


# ==========================================================
# Prediction Endpoint
# ==========================================================

@app.post("/predict")
def predict(student: StudentData):

    # Create DataFrame with EXACT training column names
    input_df = pd.DataFrame([{
        "Marital status": student.Marital_status,
        "Application mode": student.Application_mode,
        "Application order": student.Application_order,
        "Course": student.Course,
        "Daytime/evening attendance\t": student.Daytime_evening_attendance,
        "Previous qualification": student.Previous_qualification,
        "Previous qualification (grade)": student.Previous_qualification_grade,
        "Nacionality": student.Nacionality,
        "Mother's qualification": student.Mothers_qualification,
        "Father's qualification": student.Fathers_qualification,
        "Mother's occupation": student.Mothers_occupation,
        "Father's occupation": student.Fathers_occupation,
        "Admission grade": student.Admission_grade,
        "Displaced": student.Displaced,
        "Educational special needs": student.Educational_special_needs,
        "Debtor": student.Debtor,
        "Tuition fees up to date": student.Tuition_fees_up_to_date,
        "Gender": student.Gender,
        "Scholarship holder": student.Scholarship_holder,
        "Age at enrollment": student.Age_at_enrollment,
        "International": student.International,
        "Curricular units 1st sem (credited)": student.Curricular_units_1st_sem_credited,
        "Curricular units 1st sem (enrolled)": student.Curricular_units_1st_sem_enrolled,
        "Curricular units 1st sem (evaluations)": student.Curricular_units_1st_sem_evaluations,
        "Curricular units 1st sem (approved)": student.Curricular_units_1st_sem_approved,
        "Curricular units 1st sem (grade)": student.Curricular_units_1st_sem_grade,
        "Curricular units 1st sem (without evaluations)": student.Curricular_units_1st_sem_without_evaluations,
        "Curricular units 2nd sem (credited)": student.Curricular_units_2nd_sem_credited,
        "Curricular units 2nd sem (enrolled)": student.Curricular_units_2nd_sem_enrolled,
        "Curricular units 2nd sem (evaluations)": student.Curricular_units_2nd_sem_evaluations,
        "Curricular units 2nd sem (approved)": student.Curricular_units_2nd_sem_approved,
        "Curricular units 2nd sem (grade)": student.Curricular_units_2nd_sem_grade,
        "Curricular units 2nd sem (without evaluations)": student.Curricular_units_2nd_sem_without_evaluations,
        "Unemployment rate": student.Unemployment_rate,
        "Inflation rate": student.Inflation_rate,
        "GDP": student.GDP
    }])

    # Scale
    scaled_data = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_data)

    # Decode prediction
    predicted_class = label_encoder.inverse_transform(prediction)

    return {
        "Prediction": predicted_class[0]
    }