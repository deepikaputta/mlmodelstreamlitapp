
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY . .

# Expose the port the app will run on
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "1.py", "--server.port=8501", "--server.address=0.0.0.0"]

