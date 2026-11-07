# src/features/engineer.py
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

# Holdout fijo: se define UNA vez aquí y lo reutilizan entrenamiento y monitoreo
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('feature-engineering')

def create_features(df):
    """Create new features from existing data."""
    logger.info("Creating new features")

    # Make a copy to avoid modifying the original dataframe
    df_featured = df.copy()

    # Calculate house age
    current_year = datetime.now().year
    df_featured['house_age'] = current_year - df_featured['year_built']
    logger.info("Created 'house_age' feature")

    # ❌ NO crear price_per_sqft = price / sqft: usa el target (price), que no existe
    #    cuando el modelo predice. Ver la sección "Fuga de datos".

    # Bedroom to bathroom ratio
    df_featured['bed_bath_ratio'] = df_featured['bedrooms'] / df_featured['bathrooms']
    # Handle division by zero
    df_featured['bed_bath_ratio'] = df_featured['bed_bath_ratio'].replace([np.inf, -np.inf], np.nan)
    df_featured['bed_bath_ratio'] = df_featured['bed_bath_ratio'].fillna(0)
    logger.info("Created 'bed_bath_ratio' feature")

    # Do NOT one-hot encode categorical variables here; let the preprocessor handle it
    return df_featured

def create_preprocessor():
    """Create a preprocessing pipeline."""
    logger.info("Creating preprocessor pipeline")

    # Define feature groups
    categorical_features = ['location', 'condition']
    numerical_features = ['sqft', 'bedrooms', 'bathrooms', 'house_age', 'bed_bath_ratio']

    # Preprocessing for numerical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))
    ])

    # Preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessors in a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return preprocessor

def run_feature_engineering(input_file, output_file, preprocessor_file):
    """Full feature engineering pipeline."""
    # Load cleaned data
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)

    # 1. Separar el holdout ANTES de transformar: el test no debe influir en nada que se "aprenda"
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    logger.info(f"Fixed split: train={len(train_df)}, test={len(test_df)}")

    # 2. Create features (solo con columnas conocidas al momento de predecir)
    train_feat = create_features(train_df)
    test_feat = create_features(test_df)
    logger.info(f"Created featured train set with shape: {train_feat.shape}")

    # 3. Ajustar el preprocessor SOLO con train (imputación y categorías del one-hot)
    preprocessor = create_preprocessor()
    preprocessor.fit(train_feat.drop(columns=['price']))
    logger.info("Fitted the preprocessor on the training split only")

    # Save the preprocessor
    Path(preprocessor_file).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, preprocessor_file)
    logger.info(f"Saved preprocessor to {preprocessor_file}")

    # 4. Transformar ambos splits; guardar con nombres legibles y la columna `split`
    parts = []
    for name, part in (('train', train_feat), ('test', test_feat)):
        X_t = preprocessor.transform(part.drop(columns=['price']))
        out = pd.DataFrame(X_t, columns=preprocessor.get_feature_names_out(), index=part.index)
        out['price'] = part['price'].values
        out['split'] = name
        parts.append(out)
    df_transformed = pd.concat(parts).sort_index()
    df_transformed.to_csv(output_file, index=False)
    logger.info(f"Saved fully preprocessed data to {output_file} (shape {df_transformed.shape})")

    return df_transformed

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Feature engineering for housing data.')
    parser.add_argument('--input', required=True, help='Path to cleaned CSV file')
    parser.add_argument('--output', required=True, help='Path for output CSV file (engineered features)')
    parser.add_argument('--preprocessor', required=True, help='Path for saving the preprocessor')

    args = parser.parse_args()

    run_feature_engineering(args.input, args.output, args.preprocessor)