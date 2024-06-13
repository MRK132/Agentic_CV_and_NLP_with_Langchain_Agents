# Use an official Python 3.11 runtime as a parent image
FROM python:3.11

# Set the working directory in the container
WORKDIR /usr/src/app

# Install gdb and unzip utility
RUN apt-get update && apt-get install -y unzip && rm -rf /var/lib/apt/lists/*

ENV MPLBACKEND=Agg

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Unzip the images file into the desired directory
RUN unzip images.zip -d images

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Run main.py when the container launches
CMD ["python3", "./main.py"]
