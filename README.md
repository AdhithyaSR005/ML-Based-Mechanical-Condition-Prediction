# ML-Based Mechanical Condition Prediction using ANSYS and Machine Learning

A machine learning pipeline developed to predict the operating condition of mechanical components using structural simulation data generated from **ANSYS**. The project includes component-specific predictive models and a **Streamlit** web application for real-time condition prediction.

---

## Overview

Structural analysis using ANSYS can be computationally expensive when repeated for multiple operating conditions. This project demonstrates how supervised machine learning can be used to learn from simulation data and instantly predict whether a mechanical component is operating under **SAFE** or **UNSAFE** conditions.

The workflow combines finite element simulation with machine learning to reduce analysis time while maintaining reliable predictions.

---

## Objectives

- Develop predictive models using ANSYS simulation data.
- Predict the operating condition of mechanical components.
- Reduce dependence on repeated structural simulations.
- Provide an interactive prediction interface using Streamlit.

---

## Components Included

The project contains predictive models for:

- Gear
- Gear Shaft
- Bearing
- Crankshaft

Each component has its own trained machine learning model built using simulation-generated datasets.

---

## Workflow

1. Generate structural simulation data using ANSYS.
2. Prepare and clean datasets.
3. Train component-specific machine learning models.
4. Save trained models using Pickle.
5. Deploy a Streamlit web application.
6. Predict component condition in real time.

---

## Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- ANSYS Mechanical
- Pickle

---

## Features

- Interactive Streamlit interface
- Component-specific prediction models
- Fast prediction without rerunning ANSYS simulations
- Modular architecture for multiple mechanical components

---

## Repository Structure

```
ML-Based-Mechanical-Condition-Prediction
│
├── app/
│   └── app.py
│
├── src/
│   ├── gear_predictive_model.py
│   ├── bearing_predictive_model.py
│   ├── crankshaft_predictive_model.py
│   └── ...
│
├── models/
│   ├── gear_model.pkl
│   ├── bearing_model.pkl
│   ├── crankshaft_model.pkl
│   └── ...
│
├── datasets/
│   ├── gear_dataset.csv
│   ├── bearing_dataset.csv
│   └── ...
│
├── requirements.txt
│
└── README.md
```

---

## Results

- Successfully trained machine learning models for multiple mechanical components.
- Enabled real-time prediction using a Streamlit application.
- Eliminated the need to rerun ANSYS simulations for every prediction.

---

## Future Improvements

- Deep Learning models
- Explainable AI (SHAP)
- Cloud deployment
- Digital Twin integration
- Predictive maintenance dashboard

---

## Author

**Adhithya S R**

Mechanical Engineering Undergraduate  
National Institute of Technology Karnataka (NITK), Surathkal

- LinkedIn: https://www.linkedin.com/in/adhithya-s-r-b2343021a/
- GitHub: https://github.com/AdhithyaSR005
