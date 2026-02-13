# Use the same dolfinx image you are already using
FROM dolfinx/lab:stable

# Set the working directory
WORKDIR /app

# Copy your code and requirements into the container
COPY . /app

# Install the extra libraries automatically
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Streamlit uses
EXPOSE 8501

# Command to run the app immediately when the container starts
ENTRYPOINT ["streamlit", "run", "thermal_app_v4.py", "--server.port=8501", "--server.address=0.0.0.0"]