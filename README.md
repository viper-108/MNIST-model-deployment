**MLflow Machine Learning Lifecycle System - MNIST Dataset**

This repository contains the setup and implementation for managing the full machine learning lifecycle using MLflow. It covers everything from training models, logging experiments, storing artifacts, to serving models and handling inference requests.

**System Overview**
The system is designed to streamline operations in a machine learning workflow which includes:

1. Experiment tracking
2. Model registration and storage
3. Model serving for inference

**The components of the system include:**

1. Client Environments: Where models are developed and initial tests are conducted.
2. MLflow Tracking Server: Central server for logging and querying experiment data.
3. Artifact Store: Storage for model artifacts.
4. Metadata Database: Database to store experiment and model metadata.
5. MLflow Model Registry: Service for model versioning and stage management.
6. Model Serving: Component that deploys models for inference.

**Architecture**

The architecture is outlined below to show the flow of data and requests through different components of the system:

+-------------------+     logs    +-----------------------+
|                   | ----------> |                       |
| Client Environments           | | MLflow Tracking Server |
|                   | <--------  |                       |
+-------------------+     stores  +-----------+-----------+
        |                                      |
        | logs                                | stores
        |                                      |
        v                                      v
+-------------------+                    +-----+-----+
|                   |                    |           |
|   Artifact Store  | <--------------->  | Metadata  |
|                   |      metadata     | Database  |
+-------------------+                    +-----+-----+
        ^                                      |
        | registers                            |
        |                                      |
        |          +-------------------+       |
        +--------> |                   |       |
                   | MLflow Model     | <-----+
                   | Registry         |
                   |                   |
                   +---------+---------+
                             |
                             | deploys
                             |
                   +---------v---------+       +-------------------+
                   |                   |       |                   |
                   | Model Serving     | ----> | Inference Requests|
                   |                   |       |                   |
                   +-------------------+       +-------------------+


**Prerequisites**

Before setting up the system, ensure the following prerequisites are met:

1. Python 3.6 or newer
2. Docker
3. Kubernetes (optional, for scalable deployment)
4. Mlflow

**Setup and Installation**

1. MLflow Tracking Server

To set up the MLflow tracking server:

mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0

2. Model Training

Use the provided Python scripts to log experiments to the MLflow server:

python train.py

3. Model Serving

Deploy the model using a Docker container:

docker build -t mlflow-model-serving .
docker run -p 9201:9201 mlflow-model-serving

**Usage**

To interact with the system, use the following commands:

1. Logging Experiments: Run the train_model.py script.

2. Viewing Experiments: Access the MLflow UI at http://localhost:5000.

3. Model Inference: Send POST requests to the model serving endpoint:

curl -X POST -H "Content-Type: application/json" -d '{"image": [...image data... ]}' http://localhost:9201/predict

