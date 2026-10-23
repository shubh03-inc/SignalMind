
from io import BytesIO
from pathlib import Path

import pandas as pd


MAX_FILE_SIZE_MB = 25
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def read_uploaded_dataset(uploaded_file):
    """
    Read an uploaded CSV or Excel file without modifying
    the project's existing datasets.
    """

    if uploaded_file is None:
        raise ValueError("Please select a file to upload.")

    filename = Path(uploaded_file.name).name
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Unsupported file type. Upload a CSV, XLSX, or XLS file."
        )

    file_bytes = uploaded_file.getvalue()

    if not file_bytes:
        raise ValueError("The uploaded file is empty.")

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"The file exceeds the {MAX_FILE_SIZE_MB} MB upload limit."
        )

    try:
        if extension == ".csv":
            try:
                dataframe = pd.read_csv(
                    BytesIO(file_bytes),
                    encoding="utf-8-sig",
                    low_memory=False,
                )
            except UnicodeDecodeError:
                dataframe = pd.read_csv(
                    BytesIO(file_bytes),
                    encoding="cp1252",
                    low_memory=False,
                )

        else:
            # Read the first worksheet of the Excel workbook.
            # XLSX uses openpyxl; XLS uses xlrd.
            engine = "openpyxl" if extension == ".xlsx" else "xlrd"

            dataframe = pd.read_excel(
                BytesIO(file_bytes),
                engine=engine,
            )

    except ImportError as exc:
        raise ValueError(
            "An Excel-reading dependency is missing. Run: "
            "python -m pip install openpyxl xlrd"
        ) from exc

    except (ValueError, OSError, UnicodeError) as exc:
        raise ValueError(
            f"Could not read '{filename}'. Check that the file "
            "is valid and is not damaged or password-protected."
        ) from exc

    except Exception as exc:
        raise ValueError(
            f"Failed to read '{filename}': "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if dataframe.empty:
        raise ValueError(
            "The file contains no data rows. "
            "Check the header and worksheet."
        )

    if len(dataframe.columns) == 0:
        raise ValueError("The file contains no columns.")

    # Clean whitespace around column names.
    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    if any(not column for column in dataframe.columns):
        raise ValueError(
            "One or more columns have blank names. "
            "Give every column a name and upload the file again."
        )

    if not dataframe.columns.is_unique:
        duplicates = (
            dataframe.columns[
                dataframe.columns.duplicated()
            ]
            .tolist()
        )

        raise ValueError(
            f"Duplicate column names found: {duplicates}. "
            "Rename the duplicate columns and upload again."
        )

    # Remove columns that contain no values at all.
    empty_columns = [
        column
        for column in dataframe.columns
        if dataframe[column].isna().all()
    ]

    if empty_columns:
        dataframe = dataframe.drop(columns=empty_columns)

    if dataframe.empty or len(dataframe.columns) == 0:
        raise ValueError(
            "The file contains no usable data after empty columns "
            "were removed."
        )

    # Normalize column names without changing the original data.
    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    return dataframe


def validate_dataset(dataframe):
    """
    Return basic data-quality findings.

    These checks identify potential problems; they do not
    prove that the dataset is correct or incorrect.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if dataframe.empty:
        raise ValueError("The dataset has no rows.")

    if len(dataframe.columns) == 0:
        raise ValueError("The dataset has no columns.")

    findings = []

    missing_cells = int(dataframe.isna().sum().sum())

    if missing_cells:
        findings.append(
            {
                "check": "Missing values",
                "count": missing_cells,
                "status": "Review",
                "details": (
                    "Some cells have missing values. "
                    "Review them before calculating metrics."
                ),
            }
        )
    else:
        findings.append(
            {
                "check": "Missing values",
                "count": 0,
                "status": "Pass",
                "details": "No missing cells detected.",
            }
        )

    duplicate_rows = int(dataframe.duplicated().sum())

    findings.append(
        {
            "check": "Duplicate rows",
            "count": duplicate_rows,
            "status": "Review" if duplicate_rows else "Pass",
            "details": (
                "Exact duplicate rows were detected."
                if duplicate_rows
                else "No exact duplicate rows detected."
            ),
        }
    )

    constant_columns = [
        column
        for column in dataframe.columns
        if dataframe[column].nunique(dropna=True) <= 1
    ]

    findings.append(
        {
            "check": "Constant columns",
            "count": len(constant_columns),
            "status": "Review" if constant_columns else "Pass",
            "details": (
                ", ".join(constant_columns)
                if constant_columns
                else "No constant columns detected."
            ),
        }
    )

    return pd.DataFrame(findings)


def profile_dataset(dataframe):
    """
    Create a column-level profile for an uploaded dataset.
    """

    profile_rows = []

    for column in dataframe.columns:
        series = dataframe[column]

        examples = (
            series.dropna()
            .astype(str)
            .head(3)
            .tolist()
        )

        profile_rows.append(
            {
                "column": str(column),
                "data_type": str(series.dtype),
                "non_missing": int(series.notna().sum()),
                "missing": int(series.isna().sum()),
                "unique_values": int(series.nunique(dropna=True)),
                "example_values": " | ".join(examples),
            }
        )

    return pd.DataFrame(profile_rows)
