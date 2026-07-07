# ===============================
# Base Image
# ===============================
FROM python:3.13-slim

# ===============================
# Working Directory
# ===============================
WORKDIR /app

# ===============================
# Copy Requirements
# ===============================
COPY requirements.txt .

# ===============================
# Install Dependencies
# ===============================
RUN pip install --no-cache-dir -r requirements.txt

# ===============================
# Copy Entire Project
# ===============================
COPY . .

# ===============================
# Expose FastAPI Port
# ===============================
EXPOSE 8000

# ===============================
# Run FastAPI
# ===============================
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]