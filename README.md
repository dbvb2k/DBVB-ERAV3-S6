# MNIST Model CI/CD Pipeline

This repository contains a CNN model for MNIST digit classification with an automated CI/CD pipeline.

## Model Architecture
- Uses less than 20,000 parameters
- Implements Batch Normalization
- Uses Dropout for regularization
- Implements Global Average Pooling
- Achieves 99.4% validation accuracy

## Local Setup

1. Clone the repository: 
git clone <repository-url>
cd <repository-name>

2. Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

3. Install dependencies:
pip install torch torchvision pytest numpy tqdm

4. Run tests:
pytest tests/

5. Train the model:
python train.py

## Project Structure

.
├── S6_model.py # Model architecture
├── train.py # Training script
├── tests/
│ └── test_model.py # Model tests
├── models/ # Saved models directory
└── .github/
└── workflows/
└── ml-pipeline.yml # GitHub Actions workflow

Trained models are saved with the naming convention:
`mnist_model_<accuracy>_<timestamp>.pth`

## GitHub Actions
The pipeline runs automatically on every push to the repository. You can view the results in the Actions tab.

To run this project locally:
1. Create the directory structure as shown in the README
2. Create a models directory to store trained models
3. Copy all the files to their respective locations
4. Follow the setup instructions in the README

Before pushing to GitHub:
1. Make sure all tests pass locally
2. Verify the model trains successfully
3. Check that the model meets all requirements:
    - Less than 20,000 parameters
    - Uses Batch Normalization
    - Uses Dropout
    - Uses GAP
    - Achieves target accuracy

When you push to GitHub, the Actions workflow will automatically:
Set up a Python environment
    - Install dependencies
    - Run all tests
    - Train the model
    - Save the trained model as an artifact
The model file will be automatically suffixed with the accuracy and timestamp when saved, making it easy to track different versions of the model.

**Sample training logs**

Log 1
![Training Log 1](./images/Acc-Scrshot1.jpg?raw=true "Training Log 1")

Log 2:
![Training Log 2](./images/Acc-Scrshot2.jpg?raw=true "Training Log 2")

Log 3:
![Training Log 2](./images/Acc-Scrshot3.jpg?raw=true "Training Log 3")

Log 4:
![Training Log 2](./images/Acc-Scrshot4.jpg?raw=true "Training Log 4")

Log 5:
![Training Log 2](./images/Acc-Scrshot5.jpg?raw=true "Training Log 5")

Log 6:
![Training Log 2](./images/Acc-Scrshot6.jpg?raw=true "Training Log 6")

