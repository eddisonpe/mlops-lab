# 🔧 Script de Producción del MLOps Engineer
import argparse
import pandas as pd
import numpy as np
import joblib
import sys
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import yaml
import logging
from mlflow.tracking import MlflowClient

# Configuración de logging profesional
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Argumentos de línea de comandos configurables."""
    parser = argparse.ArgumentParser(description="Train and register final model from config.")
    parser.add_argument("--config", type=str, required=True, help="Path to model_config.yaml")
    parser.add_argument("--data", type=str, required=True, help="Path to processed CSV dataset")
    parser.add_argument("--models-dir", type=str, required=True, help="Directory to save trained model")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None, help="MLflow tracking URI")
    return parser.parse_args()

def get_model_instance(name, params):
    """Factory para instanciar modelos desde configuración."""
    model_map = {
        'LinearRegression': LinearRegression,
        'RandomForest': RandomForestRegressor,
        'GradientBoosting': GradientBoostingRegressor,
        'HistGradientBoosting': HistGradientBoostingRegressor,
    }
    if name not in model_map:
        raise ValueError(f"Unsupported model: {name}")
    return model_map[name](**params)

def load_and_split_data(data_path, config):
    """Carga y divide datos usando configuración."""
    logger.info(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)

    target = config['model']['target_variable']

    # Holdout fijo: se usa la columna `split` creada en feature engineering (guía 2).
    # No se vuelve a dividir aquí: así todos los modelos se evalúan sobre las mismas 16 casas
    # y el preprocessor nunca vio el test.
    train = data[data['split'] == 'train']
    test = data[data['split'] == 'test']

    features = [c for c in data.columns if c not in (target, 'split')]
    logger.info(f"Using {len(features)} features")

    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    logger.info(f"Data split: Train {X_train.shape}, Test {X_test.shape}")
    return X_train, X_test, y_train, y_test

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    """Entrena y evalúa el modelo."""
    logger.info("Training model...")
    model.fit(X_train, y_train)

    # Predicciones y métricas
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    mape = mean_absolute_percentage_error(y_test, y_pred)   # error relativo: 0.05 = 5 %

    # Baseline: predecir siempre la mediana de train. Un modelo que no le gana no sirve,
    # por muy bueno que parezca el R².
    baseline = DummyRegressor(strategy="median").fit(X_train, y_train)
    baseline_mae = mean_absolute_error(y_test, baseline.predict(X_test))

    metrics = {
        'mae': float(mae),
        'r2': float(r2),
        'rmse': float(rmse),
        'mape': float(mape),
        'baseline_mae': float(baseline_mae),
    }

    logger.info(f"Model performance - R²: {r2:.4f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}, "
                f"MAPE: {mape:.4f} (baseline MAE {baseline_mae:.2f})")
    return model, metrics

def check_quality_gate(metrics, config):
    """Compara las métricas del holdout contra `model.quality_gate` del YAML.

    Devuelve la lista de incumplimientos (vacía = pasa). Umbrales opcionales:
      min_r2:             R² mínimo en el holdout
      max_mape:           error porcentual máximo
      must_beat_baseline: MAE debe ser menor que predecir la mediana
    """
    gate = config['model'].get('quality_gate', {}) or {}
    failures = []
    if 'min_r2' in gate and metrics['r2'] < gate['min_r2']:
        failures.append(f"R² {metrics['r2']:.4f} < min_r2 {gate['min_r2']}")
    if 'max_mape' in gate and metrics['mape'] > gate['max_mape']:
        failures.append(f"MAPE {metrics['mape']:.4f} > max_mape {gate['max_mape']}")
    if gate.get('must_beat_baseline', False) and metrics['mae'] >= metrics['baseline_mae']:
        failures.append(f"MAE {metrics['mae']:.0f} no mejora el baseline (mediana) {metrics['baseline_mae']:.0f}")
    return failures

def log_failed_gate(config, metrics, failures, mlflow_tracking_uri):
    """Deja constancia en MLflow del intento rechazado (sin registrar el modelo)."""
    if not mlflow_tracking_uri:
        return
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("House Price Prediction - Production")
    with mlflow.start_run(run_name="quality_gate_failed"):
        mlflow.log_params(config['model']['parameters'])
        mlflow.log_metrics(metrics)
        mlflow.set_tag("quality_gate", "failed")
        mlflow.set_tag("quality_gate_failures", " | ".join(failures))

def save_model_artifacts(model, config, metrics, models_dir):
    """Guarda modelo y artefactos."""
    import os
    os.makedirs(models_dir, exist_ok=True)

    # Guardar modelo entrenado
    model_path = os.path.join(models_dir, 'trained/house_price_model.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    # Actualizar configuración con métricas finales
    config['model']['final_metrics'] = {
        'mae': float(metrics['mae']),
        'r2': float(metrics['r2']),
        'rmse': float(metrics['rmse'])
    }

    # Guardar configuración actualizada
    config_path = os.path.join(models_dir, 'trained/house_price_model.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    logger.info(f"✅ Model saved to {model_path}")
    logger.info(f"✅ Config saved to {config_path}")

    return model_path, config_path

def register_model_in_mlflow(model, config, metrics, mlflow_tracking_uri, X_sample=None):
    """Registra modelo en MLflow con metadatos completos.

    X_sample: DataFrame con algunas filas de entrada reales. Se usa para inferir la
    firma del modelo (nombres y tipos de columnas) y guardar un input_example:
    así MLflow valida en inferencia que lleguen las mismas features.
    """
    if not mlflow_tracking_uri:
        logger.info("No MLflow tracking URI provided, skipping MLflow logging")
        return None

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("House Price Prediction - Production")

    model_name = config['model']['name']

    with mlflow.start_run(run_name="production_training"):
        # Log parámetros del modelo
        mlflow.log_params(config['model']['parameters'])

        # Log métricas de rendimiento
        mlflow.log_metrics(metrics)

        # Log información adicional
        mlflow.log_param("total_features",
                        config['model']['feature_sets']['total_features'])
        mlflow.log_param("holdout", "split fijo 80/20 (random_state=42) de feature engineering")
        gate = config['model'].get('quality_gate', {}) or {}
        for k, v in gate.items():
            mlflow.log_param(f"gate_{k}", v)
        mlflow.set_tag("quality_gate", "passed")   # si llegó aquí, pasó el gate de main()

        # Registrar modelo con firma inferida de datos reales.
        # infer_signature(None, None) no es válido: necesita al menos inputs u outputs.
        signature, input_example = None, None
        if X_sample is not None and len(X_sample) > 0:
            input_example = X_sample.head(5)
            signature = mlflow.models.infer_signature(input_example, model.predict(input_example))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name,
            signature=signature,
            input_example=input_example,
        )

        # Transicionar a producción: el umbral es el mismo min_r2 del quality gate
        # (default 0.8 si el YAML no define gate)
        min_r2 = (config['model'].get('quality_gate') or {}).get('min_r2', 0.8)
        if metrics['r2'] >= min_r2:
            client = MlflowClient()
            model_version = client.get_latest_versions(model_name, stages=["None"])[0]
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Production"
            )
            logger.info(f"✅ Model promoted to Production stage in MLflow")

        run_id = mlflow.active_run().info.run_id
        logger.info(f"✅ Model logged to MLflow with run_id: {run_id}")
        return run_id

def main():
    """Pipeline principal de entrenamiento."""
    args = parse_args()

    # Cargar configuración del modelo
    logger.info(f"Loading config from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Cargar y preparar datos
    X_train, X_test, y_train, y_test = load_and_split_data(args.data, config)

    # Instanciar modelo desde configuración
    model_name = config['model']['best_model']
    model_params = config['model']['parameters']
    model = get_model_instance(model_name, model_params)
    logger.info(f"Created {model_name} model with parameters: {model_params}")

    # Entrenar y evaluar
    trained_model, metrics = train_and_evaluate_model(
        model, X_train, y_train, X_test, y_test
    )

    # Quality gate (reto lab 13.2): si el modelo no cumple los umbrales del YAML,
    # NO se guarda ni se registra; el proceso termina con exit 1, el job de CI falla
    # y build-and-publish no corre.
    failures = check_quality_gate(metrics, config)
    if failures:
        for f in failures:
            logger.error(f"❌ Quality gate: {f}")
        log_failed_gate(config, metrics, failures, args.mlflow_tracking_uri)
        logger.error("Model NOT saved or registered. Exiting with code 1.")
        sys.exit(1)
    gate = config['model'].get('quality_gate') or {}
    logger.info(f"✅ Quality gate passed ({gate if gate else 'no thresholds defined'})")

    # Guardar artefactos localmente
    model_path, config_path = save_model_artifacts(
        trained_model, config, metrics, args.models_dir
    )

    # Registrar en MLflow
    run_id = register_model_in_mlflow(
        trained_model, config, metrics, args.mlflow_tracking_uri, X_sample=X_test
    )

    logger.info("🚀 Training pipeline completed successfully!")

    return {
        'model_path': model_path,
        'config_path': config_path,
        'metrics': metrics,
        'mlflow_run_id': run_id
    }

if __name__ == "__main__":
    main()