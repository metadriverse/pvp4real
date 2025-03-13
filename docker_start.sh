#!/bin/bash
set -e

# Source the ROS environment
source /opt/ros/foxy/setup.bash

# Optional: Source additional workspaces if necessary
# source /app/your_workspace/install/setup.bash

# Print an initialization message
echo "Docker container initialized successfully. Listing active ROS topics:"

# List active ROS topics
ros2 topic list

# Inform the user they're now in an interactive shell
echo "Entering interactive shell with ROS2 environment. Enjoy!"

# Replace the current process with a Bash shell so that the container stays running interactively
exec bash