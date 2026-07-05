"""Great Expectations validation — exact rules from the notebook."""
from __future__ import annotations

import great_expectations as gx
import pandas as pd

from config.config import (
    CATEGORY_RULES,
    DTYPE_RULES,
    EXPECTED_COLUMNS,
    NULL_RULES,
    NUMERICAL_RULES,
    UNIQUE_RULES,
)
from utils.logger import get_logger

logger = get_logger(__name__)


def validate(df: pd.DataFrame) -> bool:
    """Run all GX expectations against *df*.

    Returns True if all pass, False otherwise.
    Logs a warning for every failed expectation.
    """
    logger.info("Starting Great Expectations validation (%d rows)", len(df))

    context = gx.get_context()
    data_source = context.data_sources.add_pandas(name="telco_source")
    data_asset = data_source.add_dataframe_asset(name="telco_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        name="telco_batch"
    )
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    failures: list[str] = []

    # 1. Expected columns
    for col in EXPECTED_COLUMNS:
        result = batch.validate(
            gx.expectations.ExpectColumnToExist(column=col)
        )
        if not result["success"]:
            failures.append(f"Missing column: {col}")

    # 2. Categorical rules
    for col, allowed in CATEGORY_RULES.items():
        result = batch.validate(
            gx.expectations.ExpectColumnValuesToBeInSet(
                column=col, value_set=allowed
            )
        )
        if not result["success"]:
            failures.append(f"Bad categories in {col}")

    # 3. Numerical range rules
    for col, rule in NUMERICAL_RULES.items():
        kwargs: dict = {"column": col, "min_value": rule["min"]}
        if rule["max"] is not None:
            kwargs["max_value"] = rule["max"]
        result = batch.validate(
            gx.expectations.ExpectColumnValuesToBeBetween(**kwargs)
        )
        if not result["success"]:
            failures.append(f"Out-of-range values in {col}")

    # 4. Null rules
    for col, allow_null in NULL_RULES.items():
        if not allow_null:
            result = batch.validate(
                gx.expectations.ExpectColumnValuesToNotBeNull(column=col)
            )
            if not result["success"]:
                failures.append(f"Unexpected nulls in {col}")

    # 5. Uniqueness rules
    for col, is_unique in UNIQUE_RULES.items():
        if is_unique:
            result = batch.validate(
                gx.expectations.ExpectColumnValuesToBeUnique(column=col)
            )
            if not result["success"]:
                failures.append(f"Duplicate values in {col}")

    if failures:
        for msg in failures:
            logger.warning("Validation FAILED: %s", msg)
        return False

    logger.info("All %d expectations passed", len(EXPECTED_COLUMNS) + len(CATEGORY_RULES) + len(NUMERICAL_RULES) + len(NULL_RULES) + len(UNIQUE_RULES))
    return True
