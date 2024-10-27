# Use the official Python image as a base
FROM python:3.11.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install the required Python packages
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the entire project into the container
COPY . /app/

# Expose the port that Streamlit runs on
EXPOSE 8501


# Set the entrypoint to run the Streamlit app
CMD ["streamlit", "run", "streamlit_chatbot_rag_nomad_foods.py"]
