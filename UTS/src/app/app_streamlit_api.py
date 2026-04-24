import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

def main():
    menu                            = st.sidebar.selectbox("Menu", ["Prediction", "Visualization"])
    if menu == 'Prediction':
        st.title("Prediksi Penempatan Kerja")
        student_id                  = st.text_input("Student ID", "1")
        gender                      = st.selectbox("Gender", ("Male", "Female"), index=0)
        ssc_percentage              = st.number_input("SSC Percentage", min_value=0.0, max_value=100.0, value=75.0)
        hsc_percentage              = st.number_input("HSC Percentage", min_value=0.0, max_value=100.0, value=78.0)
        degree_percentage           = st.number_input("Degree Percentage", min_value=0.0, max_value=100.0, value=80.0)
        cgpa                        = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.5)
        entrance_exam_score         = st.number_input("Entrance Exam Score", min_value=0.0, max_value=100.0, value=70.0)
        technical_skill_score       = st.slider("Technical Skill Score", min_value=0.0, max_value=100.0, value=75.0)
        soft_skill_score            = st.slider("Soft Skill Score", min_value=0.0, max_value=100.0, value=70.0)
        internship_count            = st.number_input("Internship Count", min_value=0, value=1)
        live_projects               = st.number_input("Live Projects", min_value=0, value=1)
        work_experience_months      = st.number_input("Work Experience (Months)", min_value=0, value=6)
        certifications              = st.number_input("Certifications", min_value=0, value=2)
        extracurricular_activities  = st.selectbox("Extracurricular Activities", ("Yes", "No"), index=0)
        attendance_percentage       = st.slider("Attendance Percentage", min_value=0.0, max_value=100.0, value=85.0)
        backlogs                    = st.number_input("Backlogs", min_value=0, value=0)

        if st.button("Make Prediction"):
            features = [[student_id, gender, ssc_percentage, hsc_percentage, degree_percentage, cgpa, entrance_exam_score, technical_skill_score, soft_skill_score, internship_count,
                        live_projects, work_experience_months, certifications, extracurricular_activities, attendance_percentage, backlogs]]
            features = pd.DataFrame(features, columns=['student_id', 'gender', 'ssc_percentage', 'hsc_percentage', 'degree_percentage', 'cgpa', 'entrance_exam_score', 
                                                    'technical_skill_score', 'soft_skill_score', 'internship_count', 'live_projects', 'work_experience_months', 'certifications', 
                                                    'extracurricular_activities', 'attendance_percentage', 'backlogs'])
            features = features.to_dict(orient="records")[0]
            result = make_prediction(features)
            if result["placement_prediction"] == 1:
                st.success("✅️ Mendapatkan Penempatan Posisi")
                st.write(f"💰 Perkiraan Gaji: {result['salary_prediction']:.2f} LPA")
            else:
                st.error("❌ Tidak Mendapatkan Penempatan Posisi")
    else:
        st.title("Visualisasi Persebaran Data Placement Status dan Salary")
        df = pd.read_csv("src/data/ingested/B.csv")
        fig, ax = plt.subplots()
        df['placement_status'].value_counts().plot(kind='bar', ax=ax)
        ax.set_title("Distribusi Placement Status")
        ax.set_xlabel("Placement Status")      
        ax.set_ylabel("Frekuensi")
        st.pyplot(fig)

        fig, ax = plt.subplots()
        df['salary_package_lpa'].hist(ax=ax)
        ax.set_title("Grafik Distribusi Salary Package")
        ax.set_xlabel("Salary Package (LPA)")      
        ax.set_ylabel("Frekuensi")
        st.pyplot(fig)

def make_prediction(features):
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=features, timeout=5)
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to FastAPI server. Start it first:\n"
                 "`uvicorn api:app --reload`\n")
        return None

if __name__ == "__main__":
    main()