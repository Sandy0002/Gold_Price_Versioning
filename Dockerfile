FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Copy the rest of the code
COPY ./src ./src
COPY ./models ./models

# Expose port 8000 (Render expects a web service)
EXPOSE 8000

# Start the FastAPI server
CMD uvicorn src.api:app --host 0.0.0.0 --port $PORT