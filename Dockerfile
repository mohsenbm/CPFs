FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app
ENV PYTHONPATH=/app

# Copy repo contents
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the Python scripts
CMD python src/experiments/cpfs_compilation.py && \
    python src/experiments/cpfs_no_compilation.py && \
    python src/experiments/hardware.py
