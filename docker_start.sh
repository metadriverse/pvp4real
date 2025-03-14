#!/bin/bash
set -e

# Set up a trap to print a farewell message when the script exits.
trap 'echo "Bye! The Docker container is exiting."' EXIT

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

# Execute the command passed as arguments (e.g., bash)
exec "$@"