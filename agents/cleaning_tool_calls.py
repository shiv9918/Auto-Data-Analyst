"""CrewAI tools scoped to one working DataFrame and one set of user choices."""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Literal

import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from tools import cleaning_tools as ct

logger = logging.getLogger("auto_data_analyst.cleaning")


class CleaningToolError(ValueError):
    """The tool call cannot be safely applied to this analysis run."""


class NoArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)


class FillArguments(NoArguments):
    strategy: Literal["auto", "mean", "median", "mode"] = Field(
        description="Use auto for numeric median and text mode."
    )


class OutlierArguments(NoArguments):
    strategy: Literal["cap", "remove"] = Field(description="Use the selected outlier strategy.")


class EncodeArguments(NoArguments):
    method: Literal["onehot", "mean"] = Field(description="Use the selected encoding method.")


class CleaningTool(BaseTool):
    _callback: Callable[..., dict] = PrivateAttr()

    def __init__(self, callback: Callable[..., dict], **kwargs):
        super().__init__(**kwargs)
        self._callback = callback

    def _run(self, **kwargs) -> str:
        arguments = self.args_schema.model_validate(kwargs)
        return json.dumps(self._callback(**arguments.model_dump()), separators=(",", ":"))


@dataclass(frozen=True)
class _CleaningStep:
    name: str
    description: str
    count_problems: Callable[[pd.DataFrame], int]
    clean: Callable[..., tuple[pd.DataFrame, object]]
    arguments: type[NoArguments] = NoArguments


def _missing_cells(frame: pd.DataFrame) -> int:
    return sum(issue["count"] for issue in ct.detect_missing_values(frame).values())


def _outlier_cells(frame: pd.DataFrame) -> int:
    return sum(issue["count"] for issue in ct.detect_outliers(frame).values())


def _empty_rows_and_columns(frame: pd.DataFrame) -> int:
    missing = frame.isna()
    return int(missing.all(axis=1).sum() + missing.all(axis=0).sum())


def _remove_empty_rows_and_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, None]:
    frame, _ = ct.remove_empty_rows(frame)
    frame, _ = ct.remove_empty_columns(frame)
    return frame, None


def _changed_rows_and_columns(before: pd.DataFrame, after: pd.DataFrame) -> tuple[int, int]:
    # Column additions/removals change the schema for every surviving row.
    changed_columns = len(before.columns.symmetric_difference(after.columns))
    if changed_columns:
        return len(before), changed_columns
    if len(before) != len(after):
        return len(before) - len(after), len(before.columns)

    changed_rows = pd.Series(False, index=before.index)
    for column in before.columns:
        old, new = before[column], after[column]
        different = old.ne(new).fillna(True) & (old.notna() | new.notna())
        if different.any():
            changed_columns += 1
            changed_rows |= different
    return int(changed_rows.sum()), changed_columns


class CleaningSession:
    def __init__(self, original: pd.DataFrame, config: dict):
        self.frame = original.copy()
        self.verified = False
        self.report_lines = [f"Loaded {len(original):,} rows x {len(original.columns):,} columns."]
        self._original = original
        self._config = config.copy()
        self._inspected = False
        self._next_step = 0
        self._pending: dict[str, int] = {}
        self._steps = (
            _CleaningStep("normalize_placeholders", "Convert recognised missing-value text into nulls.",
                          ct.count_placeholder_values, ct.handle_placeholder_missing_values),
            _CleaningStep("remove_empty_rows_and_columns", "Drop fully empty rows, then fully empty columns.",
                          _empty_rows_and_columns, _remove_empty_rows_and_columns),
            _CleaningStep("remove_duplicates", "Drop exact duplicate rows before imputing missing values.",
                          ct.detect_duplicate_rows, ct.remove_duplicate_rows),
            _CleaningStep("fill_missing", "Fill the current missing cells with the specified strategy.",
                          _missing_cells, ct.fill_missing_values, FillArguments),
            _CleaningStep("trim_whitespace", "Trim leading, trailing and repeated spaces in text.",
                          ct.count_extra_space_values, ct.remove_extra_spaces),
            _CleaningStep("handle_outliers", "Treat current IQR outliers with the user's selected strategy.",
                          self._actionable_outliers, ct.handle_outliers, OutlierArguments),
            _CleaningStep("encode_categorical", "Encode eligible categories using the user's options.",
                          self._encodable_columns, self._encode, EncodeArguments),
        )

    def _actionable_outliers(self, frame: pd.DataFrame) -> int:
        if self._config["outlier_strategy"] in ("cap", "remove"):
            return _outlier_cells(frame)
        return 0

    def _encodable_columns(self, frame: pd.DataFrame) -> int:
        return len(ct.encodable_columns(
            frame, self._config["encoding_method"], self._config["target_column"], self._config["max_categories"]
        ))

    def _encode(self, frame: pd.DataFrame, *, method: str) -> tuple:
        return ct.standardize_categorical_values(
            frame, method=method, target_column=self._config["target_column"],
            max_categories=self._config["max_categories"],
        )

    def _refresh_pending(self) -> None:
        self._pending = {}
        for step in self._steps[self._next_step:]:
            count = step.count_problems(self.frame)
            if count:
                self._pending[step.name] = count

    def inspect(self) -> dict:
        self._refresh_pending()
        self._inspected = True
        result = {
            "shape": self.frame.shape,
            "numeric_columns": len(self.frame.select_dtypes(include="number").columns),
            "outlier_strategy": self._config["outlier_strategy"],
            "encoding_method": self._config["encoding_method"],
            "max_categories": self._config["max_categories"],
            "pending": self._pending.copy(),
        }
        if self._config["outlier_strategy"] == "none":
            result["outlier_cells_to_keep"] = _outlier_cells(self.frame)
        logger.info("Cleaning tool inspect_dataset input={} result=%s", result)
        return result

    def apply(self, index: int, **arguments) -> dict:
        step = self._steps[index]
        if not self._inspected or self.verified:
            raise CleaningToolError("Cleaning tools must run after inspection and before verification.")
        if index < self._next_step:
            raise CleaningToolError("This cleaning step was already applied or passed.")
        if step.name == "handle_outliers" and arguments["strategy"] != self._config["outlier_strategy"]:
            raise CleaningToolError("The requested outlier strategy conflicts with the user's choice.")
        if step.name == "encode_categorical" and arguments["method"] != self._config["encoding_method"]:
            raise CleaningToolError("The requested encoding method conflicts with the user's choice.")
        for earlier in self._steps[self._next_step:index]:
            if earlier.name in self._pending:
                raise CleaningToolError(f"Resolve {earlier.name} before calling {step.name}.")

        before = self.frame
        affected_rows = affected_columns = 0
        if step.name in self._pending:
            cleaned, _ = step.clean(before.copy(), **arguments)
            affected_rows, affected_columns = _changed_rows_and_columns(before, cleaned)
            self.frame = cleaned
        self._next_step = index + 1
        self._refresh_pending()

        result = {
            "shape_before": before.shape, "shape_after": self.frame.shape,
            "affected_rows": affected_rows, "affected_columns": affected_columns,
            "rows_removed": len(before) - len(self.frame),
            "pending": self._pending.copy(),
        }
        logger.info("Cleaning tool %s input=%s result=%s", step.name, arguments, result)
        if affected_rows or affected_columns:
            label = step.name.replace("_", " ").capitalize()
            self.report_lines.append(f"{label} affected {affected_rows:,} rows and {affected_columns:,} columns.")
        return result

    def verify(self) -> dict:
        if not self._inspected:
            raise CleaningToolError("Inspect the dataset before verification.")
        if self._pending:
            raise CleaningToolError("Dataset verification found unfinished cleaning steps.")
        ct.validate_cleaned_dataset(self._original, self.frame)
        if _missing_cells(self.frame):
            raise CleaningToolError("Dataset verification found unfilled missing values.")
        self.verified = True
        result = {
            "verified": True, "shape": self.frame.shape,
            "remaining_duplicate_rows": ct.detect_duplicate_rows(self.frame),
        }
        logger.info("Cleaning tool verify_dataset input={} result=%s", result)
        return result

    def tools(self) -> list[BaseTool]:
        def tool(name: str, description: str, callback: Callable, schema: type[BaseModel] = NoArguments):
            # Usage limits also tell CrewAI to execute stateful calls sequentially.
            return CleaningTool(name=name, description=description, callback=callback,
                                args_schema=schema, max_usage_count=1)

        return [
            tool("inspect_dataset", "Inspect dataset shape, problem counts and selected settings. Call first.", self.inspect),
            *(tool(step.name, step.description, partial(self.apply, index), step.arguments)
              for index, step in enumerate(self._steps)),
            tool("verify_dataset", "Verify the cleaned dataset after pending steps are resolved. Call last.", self.verify),
        ]
