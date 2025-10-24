FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install PDM for package management
RUN pip install --no-cache-dir pdm

# Copy dependency definition files first to leverage Docker cache
COPY pyproject.toml pdm.lock ./

# Install project dependencies using PDM for production
RUN pdm sync --prod

# Copy the rest of the application source code
COPY . .

# Expose the port the app will run on
EXPOSE 8000

# Define the command to run the application
CMD ["pdm", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]