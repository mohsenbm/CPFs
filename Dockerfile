FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy repo contents
COPY . .

# Install dependencies (if you have requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt || true

# Run the Python scripts
CMD python src/experiments/cpfs_compilation.py && \
    python src/experiments/cpfs_no_compilation.py && \
    python src/experiments/hardware.py