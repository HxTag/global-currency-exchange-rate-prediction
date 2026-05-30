@echo off

echo Starting Currency Prediction Project...

echo.
echo Starting FastAPI server...

start cmd /k uvicorn api.app:app --reload

echo.
echo Starting Streamlit Dashboard...

start cmd /k streamlit run dashboard/streamlit_app.py

echo.
echo Project started successfully!
echo Dashboard: http://localhost:8501
echo API Docs: http://127.0.0.1:8000/docs
pause