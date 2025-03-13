# Use the official ROS Foxy Desktop image as a base (Ubuntu 20.04)
FROM osrf/ros:foxy-desktop

# Set an environment variable for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive

# Update package lists and install additional packages (e.g., pip and other tools)
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    vim \
    ros-foxy-rmw-cyclonedds-cpp \
    ros-foxy-rosidl-generator-dds-idl \
    && rm -rf /var/lib/apt/lists/*

# Optionally, copy your Python requirements file and install dependencies
# COPY requirements.txt /tmp/requirements.txt
# RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Create a dedicated directory for your application code
WORKDIR /app

# Copy your docker_start.sh script into /app
COPY docker_start.sh .

# Ensure that docker_start.sh is executable
RUN chmod +x docker_start.sh

# Expose any ports your application uses (if applicable)
EXPOSE 8080

# Set the entrypoint to start your application
CMD ["./docker_start.sh"]
