#!/bin/bash


# Step 0: checking the entered argument
num_dataset=$1
if ! [[ $num_dataset =~ ^[0-9]+$ ]] || [ $# -eq 0 ]; 
then
    echo "If you want multiple datasets, then specify the first parameter as a number!"
    num_dataset=1
else
    if [ $num_dataset -qt 5 ]; then
        echo "The number of datasets must be less than 5!"
        num_dataset=5
    fi
fi


# Step 1: Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "Python 3 is not installed. Please install Python 3 to proceed."
    exit 1
fi

# Step 2: Check if virtual environment is present. If not, create one.
if [ ! -d ".venv" ]
then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Step 3: Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Step 4: Install dependencies from requirements.txt
if [ -f "requirements.txt" ]
then
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found. Skipping dependency installation."
fi

# Step 5: Run Python script data_creation.py
python src/data_creation.py

# Step 6: Run Python script model_preprocessing.py
python src/model_preprocessing.py

# Step 7: Run Python script model_preparation.py
python src/model_preparation.py

# Step 8: Run Python script model_testing.py
python src/model_testing.py

# Step 9: Deactivate virtual environment
deactivate
