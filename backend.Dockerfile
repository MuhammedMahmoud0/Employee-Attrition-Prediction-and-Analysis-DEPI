FROM python:3.12-slim

# Dependencies required for numpy, pandas, statsmodels, sklearn

WORKDIR /app

COPY requirements_backend.txt .
RUN pip install --no-cache-dir -r requirements_backend.txt

# Copy ONLY backend-related files (from /app and /models)
COPY app/api.py ./api.py
COPY models ./models

# Expose FastAPI port
EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
