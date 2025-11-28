FROM python:3.12-slim

# Dependencies required for numpy, pandas, statsmodels, sklearn

WORKDIR /work

COPY requirements_backend.txt .
RUN pip install --no-cache-dir -r requirements_backend.txt

# Copy ONLY backend-related files (from /app and /models)
COPY app ./app
COPY models ./models

# Expose FastAPI port
EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
