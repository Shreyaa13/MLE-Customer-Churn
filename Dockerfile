# Use the official Apache Airflow image (adjust the version as needed)
FROM apache/airflow:2.6.1

# Switch to root to install additional packages
USER root

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install Java and create proper directory structure
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-17-jdk-headless procps bash && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /bin/bash /bin/sh && \
    mkdir -p /usr/lib/jvm/java-17-openjdk-amd64/bin
    #  && \
    # ln -sf $(which java) /usr/lib/jvm/java-17-openjdk-amd64/bin/java && \
    # ln -sf $(which javac) /usr/lib/jvm/java-17-openjdk-amd64/bin/javac

# Set JAVA_HOME and Spark environment variables
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3
ENV SPARK_HOME=/home/airflow/.local/lib/python3.7/site-packages/pyspark
ENV JAVA_OPTS="-Xmx2g -XX:+UseG1GC"

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Switch to the airflow user before installing Python dependencies
USER airflow

# Install Python dependencies using requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create a volume mount point for notebooks
VOLUME /app
