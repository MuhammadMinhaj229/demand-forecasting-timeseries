FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# We default to the FastAPI backend, but this image can be used for Streamlit too by overriding the command
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
