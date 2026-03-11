FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files (including app/, model/, streamlit_app.py, and mlflow.db)
COPY . .

# Expose the API port and the Streamlit port
EXPOSE 8000
EXPOSE 8501

# This CMD will be the default, but Docker Compose can override it
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]