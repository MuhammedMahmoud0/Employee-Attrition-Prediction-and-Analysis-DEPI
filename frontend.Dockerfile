FROM python:3.12-slim



WORKDIR /work

COPY requirements_frontend.txt .
RUN pip install --no-cache-dir -r requirements_frontend.txt

# Copy all frontend UI files
COPY app ./app
COPY data ./data  
EXPOSE 8501

CMD ["streamlit", "run", "app/entry.py", "--server.port=8501", "--server.address=0.0.0.0"]
