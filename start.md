# Terminal 1
mlflow server --host 127.0.0.1 --port 8080

# Terminal 2
uvicorn app.api.main:app --port 8000

# Terminal 3
cd app/web && npm run dev



MetricValidationTestMAE3.623.65RMSE10.4610.89R²0.96620.9627MAPE42.68%41.38%MBE-0.0560.133