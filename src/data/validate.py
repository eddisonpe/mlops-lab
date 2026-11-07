"""Gate de validación de datos: termina con código 1 si los datos no cumplen el contrato.

Uso:
    python src/data/validate.py --input data/processed/cleaned_house_data.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contract import check_contract  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("data-validation")


def main():
    parser = argparse.ArgumentParser(description="Validate a house-price dataset against the data contract.")
    parser.add_argument("--input", required=True, help="CSV to validate")
    parser.add_argument("--no-target", action="store_true", help="the CSV has no price column (inference batch)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    violations = check_contract(df, require_target=not args.no_target)

    if not violations:
        logger.info(f"✅ {args.input}: {len(df)} filas cumplen el data contract")
        return 0

    logger.error(f"❌ {args.input}: {len(violations)} violación(es) del data contract")
    for v in violations:
        logger.error(f"   [{v['check']}] {v['column']}: {v['rows']} fila(s) {v['detail']}")
    return 1


if __name__ == "__main__":
    sys.exit(main())