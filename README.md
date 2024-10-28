# Image Classification API with FastAPI

## Project Overview
This project demonstrates a Logistic Regression model built using PyTorch, served through a FastAPI application. The model predicts digits (0-9) based on images provided by the user through a simple web interface. Users can upload an image, and the model returns the digit prediction.

This project is a REST API built using FastAPI for classifying images using a pre-trained Logistic Regression model. The model predicts the class of an image, such as those from the MNIST dataset.

## Running the Code Locally

### Prerequisites
1. Python 3.8 or higher
2. torch for model training and inference
3. fastapi for building the API
4. uvicorn for running the FastAPI server
5. pillow for image processing
6. python-multipart for handling file uploads

## Features
- Upload an image (28x28 grayscale) to get a prediction.
- Model trained with PyTorch and served using FastAPI.

## Prerequisites
- Python 3.8+
- `pip` (Python package manager)

## URL for the wandb report
[https://api.wandb.ai/links/rudra-corteva-corteva-agriscience/7egoec2i](https://api.wandb.ai/links/rudra-corteva-corteva-agriscience/nwi1e6ud)

## URL for the Deployed Project
https://mlops-assignment-l6t3.onrender.com/

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/RudraCorteva/MLOps_Assignment.git
cd MLOps_Assignment
```
### 2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### 3.Install the required dependencies:
```bash
pip install -r requirements.txt
```
## Running the API Locally
1. Make sure the model.pth file (pre-trained PyTorch model) is in the project directory.
2. Start the FastAPI server:
```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```
3. Open your web browser and navigate to:
```bash
http://127.0.0.1:8000/
```
## Building and Running the Docker Container

### Prerequisites
Docker installed on your machine
### Building the Docker Image
1. Build the Docker image using the provided Dockerfile:
```bash
docker build -t logistic-regression-api .
```
2. Run the Docker container:
```bash
docker run -d -p 8000:8000 logistic-regression-api

```
3. Access the API in your browser at:
```bash
http://localhost:8000/
```

   

