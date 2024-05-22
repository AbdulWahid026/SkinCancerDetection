# Skin Cancer Detection Project

## Overview
This project focuses on detecting skin cancer by analyzing images of skin lesions. We have extracted various features from the images and stored them in a CSV file. Multiple machine learning models were trained using this dataset, and the best-performing model was saved for future use.


## Requirements
- Python 3.7+
- Required Python packages are listed in `requirements.txt`.

To install the required packages, run:
```bash
pip install -r requirements.txt
```

## Data
- `Resized_Images`: Directory containing images of skin lesions.
- `Features_Cancer.csv`: CSV file containing extracted features from the images.

## Feature Extraction
The features were extracted from the images using various image processing techniques. The following features were included:
- Color histogram
- Texture features (e.g., Local Binary Patterns)
- Shape descriptors (e.g., edge detection)
- Statistical features (e.g., mean, variance)

The extracted features were stored in `data/features.csv` with the following columns:
- `image_id`: Unique identifier for each image
- `feature1`: First feature
- `feature2`: Second feature
- ...
- `label`: Ground truth label (e.g., benign, malignant)

## Machine Learning Models
We trained multiple machine learning models on the extracted features:
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Gradient Boosting
- Neural Network

Each model was evaluated using cross-validation, and performance metrics such as accuracy, precision, recall, and F1-score were recorded.

## Best Model
The best-performing model was selected based on the evaluation metrics. In this project, the best model was found to be the `Random Forest`. This model was saved for future use.

To load the best model:
```python
import joblib
model = joblib.load('models/best_model_SVC.pkl')
```

## Usage
1. **Feature Extraction**:
    - To extract features from new images, use the script `scripts/extract_features.py`.
    - Example:
      ```bash
      python scripts/extract_features.py --input_dir new_images/ --output_file new_features.csv
      ```

2. **Training**:
    - To train the models, use the notebook `notebooks/train_models.ipynb` or the script `scripts/train_models.py`.
    - Example:
      ```bash
      python scripts/train_models.py --input_file data/features.csv --output_dir models/
      ```

3. **Evaluation**:
    - To evaluate the models, use the notebook `notebooks/evaluate_models.ipynb` or the script `scripts/evaluate_models.py`.
    - Example:
      ```bash
      python scripts/evaluate_models.py --model_file models/best_model.pkl --test_file data/test_features.csv
      ```

## Results
- The results of the evaluation are in the notebook along with performance metrics.
