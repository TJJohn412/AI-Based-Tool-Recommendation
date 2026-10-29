# Smart Machining Assistant

Smart Machining Assistant is an AI-based decision-support system for recommending suitable machining tools and process parameters based on material, hardness, machine capability, operation type, and surface requirements.

## Project status

This repository is now in a working prototype stage: the project structure is complete, the dataset is generated, the Streamlit app is running, and a trained recommendation model is saved in the `models/` folder.

## Core stack

- Python 3.12
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Plotly
- Pytest
- Git and GitHub

## Planned folder structure

- `app/` for the dashboard UI and styling
- `dataset/` for source, processed, and documentation data
- `docs/` for architecture, testing, and design records
- `models/` for trained artifacts
- `src/data/` for data generation and validation
- `src/features/` for preprocessing and encoding
- `src/training/` for training and evaluation logic
- `src/prediction/` for recommendation logic
- `src/utils/` for engineering constants and helper functions
- `tests/` for unit and validation tests

## Environment setup

Create the local Python environment and install dependencies inside the repository:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the Streamlit app from the project root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\app.py
```

Or use the repository launcher:

```powershell
python .\run_app.py
```

## Stage progress

- Stage 1: repository inspection complete
- Stage 2: project scaffold and environment configuration complete
- Stage 3: synthetic dataset generation complete
- Stage 4: engineering-rule recommendation engine complete
- Stage 5: Streamlit dashboard and model training pipeline complete

## Training the model

```powershell
.\.venv\Scripts\python.exe -m src.training.train_model
```

## Notes

This project remains modular so that each engineering task, model, and UI component can be tested independently before integration. The recommendation engine intentionally distinguishes between ML prediction, rule-based engineering logic, and demonstration-level synthetic data. Recommendation outputs also carry confidence and agreement indicators so the app can surface when the model and rule checks are aligned or when the result should be reviewed.
