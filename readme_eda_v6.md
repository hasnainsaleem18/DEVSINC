## Overview

This project performs exploratory data analysis (EDA) and feature engineering on an advertiser dataset. The main goal is to process and transform advertiser data to create features that can be used for building a recommendation system. The system aims to recommend optimal postal codes for advertisers based on their industry, campaign goals, and other characteristics.

## Prerequisites

### Required Python Libraries

Before running this notebook, ensure you have the following libraries installed:

```bash
pip install pandas numpy scikit-learn ast re scipy
```

### Library Descriptions

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning preprocessing and vectorization tools
- **ast**: Abstract Syntax Trees (for parsing string representations of Python literals)
- **re**: Regular expressions for text processing
- **scipy**: Scientific computing (used for sparse matrices)

## Data Requirements

The notebook expects a CSV file named `updated_aggregated_advertisers_dataset_v2.csv` in the same directory. This file should contain the following columns:

- `unique_goals`: Campaign goals data (likely in list format)
- `order_goals`: Order-specific goals data
- `postal_codes`: Geographic postal code data
- `unique_industries`: Industry categories
- `avg_order_duration_days`: Average duration of orders in days
- `states`: State information
- `cities`: City information
- `locations`: Location data
- `favorite_city`: Preferred city data
- `favorite_state`: Preferred state data
- `advertiser_city`: Advertiser's city
- `advertiser_state_code`: Advertiser's state code
- `advertiser_postal_code`: Advertiser's postal code

## Project Structure

### 1. Data Loading and Initial Setup

The notebook begins by importing necessary libraries and loading the dataset:

```python
import pandas as pd
import ast
import re
import numpy as np
# ... other imports

df = pd.read_csv('updated_aggregated_advertisers_dataset_v2.csv')
```

### 2. Data Cleaning

#### Goals Data Cleaning
The `clean_messy_list_string()` function processes the `unique_goals` and `order_goals` columns:
- Handles missing values
- Converts text to uppercase
- Extracts words using regular expressions
- Returns cleaned lowercase words

#### Postal Codes Processing
Uses `ast.literal_eval()` to convert string representations of lists into actual Python lists for the `postal_codes` column.

### 3. Data Analysis

The notebook includes analysis to compare `order_goals` and `unique_goals` columns:
- Calculates percentage of identical vs different rows
- Shows that 100% of rows are identical between these columns

### 4. Feature Engineering

#### Bag of Words (BOW) Vectorization

The `create_semantic_bow_vectors()` function creates three types of feature vectors:

1. **Geographic BOW Vector** (200 features)
   - Processes: states, cities, favorite_city, favorite_state, advertiser_city, advertiser_state_code
   
2. **Goals BOW Vector** (5 features)
   - Processes: unique_goals
   
3. **Industry BOW Vector** (30 features)
   - Processes: unique_industries

#### Text Preprocessing
The `preprocess_text_preserve_multiword()` function:
- Removes brackets and special characters
- Handles multi-word terms by replacing spaces with underscores
- Splits and cleans text data

#### Advanced Transformations

1. **StandardScaler**: Normalizes the `avg_order_duration_days` column
2. **MultiLabelBinarizer**: Transforms postal codes into binary features
3. **OneHotEncoder**: Encodes advertiser postal codes

### 5. Feature Combination

The final step combines all processed features into a single feature matrix:
- Scaled duration data
- Postal code binary features
- Advertiser postal code one-hot encoded features
- BOW vectors for geographic, goals, and industry data

## Code Sections Explained

### Section 1: Library Imports
Sets up all necessary tools for data processing and machine learning.

### Section 2: Data Loading
Loads the advertiser dataset from CSV file.

### Section 3: Goals Data Cleaning
Cleans and standardizes the goals columns using regex pattern matching.

### Section 4: Postal Code Processing
Converts string representations of postal code lists into proper Python lists.

### Section 5: Data Quality Check
Analyzes the similarity between order_goals and unique_goals columns.

### Section 6: BOW Vectorization
Creates three separate bag-of-words feature matrices for different types of data:
- Geographic information
- Campaign goals
- Industry categories

### Section 7: Advanced Preprocessing
Applies various sklearn transformers to create additional features:
- Standardizes numerical data
- Creates binary encodings for postal codes
- One-hot encodes categorical variables

### Section 8: Feature Matrix Assembly
Combines all processed features into a final feature matrix with 36,836 columns.

## File Dependencies

- `updated_aggregated_advertisers_dataset_v2.csv`: Main dataset file
- `eda_v6.ipynb`: Main notebook file

Make sure both files are in the same directory for the notebook to run successfully.