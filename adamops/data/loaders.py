"""
AdamOps Data Loaders Module

Provides comprehensive data loading capabilities from various sources:
- CSV files with auto-encoding detection
- Excel files (.xlsx, .xls)
- JSON files
- SQL databases (SQLite, PostgreSQL, MySQL)
- API/URL endpoints
- Compressed files (.zip, .gz)
"""

import os
import io
import json
import gzip
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import pandas as pd
import numpy as np

try:
    import chardet
    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from sqlalchemy import create_engine, text
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

from adamops.utils.logging import get_logger
from adamops.utils.helpers import ensure_dir

logger = get_logger(__name__)


# =============================================================================
# Encoding Detection
# =============================================================================

def detect_encoding(filepath: Union[str, Path], sample_size: int = 10000) -> str:
    """
    Detect the encoding of a file.
    
    Args:
        filepath: Path to the file.
        sample_size: Number of bytes to sample for detection.
    
    Returns:
        str: Detected encoding (e.g., 'utf-8', 'latin-1').
    
    Example:
        >>> encoding = detect_encoding("data.csv")
        >>> print(encoding)
        'utf-8'
    """
    if not CHARDET_AVAILABLE:
        logger.warning("chardet not available, defaulting to utf-8")
        return "utf-8"
    
    with open(filepath, "rb") as f:
        raw_data = f.read(sample_size)
    
    result = chardet.detect(raw_data)
    encoding = result.get("encoding", "utf-8")
    confidence = result.get("confidence", 0)
    
    logger.debug(f"Detected encoding: {encoding} (confidence: {confidence:.2%})")
    
    # Fall back to utf-8 if detection is uncertain
    if confidence < 0.5:
        encoding = "utf-8"
    
    return encoding or "utf-8"


# =============================================================================
# CSV Loading
# =============================================================================

def load_csv(
    filepath: Union[str, Path],
    encoding: Optional[str] = None,
    auto_detect_encoding: bool = True,
    sep: str = ",",
    header: Union[int, List[int], str] = "infer",
    index_col: Optional[Union[int, str, List]] = None,
    usecols: Optional[List] = None,
    dtype: Optional[Dict] = None,
    parse_dates: Optional[Union[bool, List]] = None,
    na_values: Optional[List] = None,
    nrows: Optional[int] = None,
    skiprows: Optional[Union[int, List]] = None,
    low_memory: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    Load data from a CSV file with auto-encoding detection.
    
    Args:
        filepath: Path to the CSV file.
        encoding: File encoding. If None and auto_detect_encoding is True, 
                  encoding will be detected automatically.
        auto_detect_encoding: Whether to auto-detect encoding.
        sep: Column separator.
        header: Row number(s) to use as column names.
        index_col: Column(s) to use as index.
        usecols: Columns to load.
        dtype: Data types for columns.
        parse_dates: Columns to parse as dates.
        na_values: Additional values to treat as NA.
        nrows: Number of rows to read.
        skiprows: Rows to skip.
        low_memory: Use low memory mode.
        **kwargs: Additional arguments passed to pd.read_csv.
    
    Returns:
        pd.DataFrame: Loaded data.
    
    Example:
        >>> df = load_csv("data.csv")
        >>> df = load_csv("data.csv", usecols=["id", "name", "value"])
        >>> df = load_csv("data.csv", parse_dates=["date_column"])
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Detect encoding if not specified
    if encoding is None and auto_detect_encoding:
        encoding = detect_encoding(filepath)
    elif encoding is None:
        encoding = "utf-8"
    
    logger.info(f"Loading CSV: {filepath} (encoding: {encoding})")
    
    try:
        df = pd.read_csv(
            filepath,
            encoding=encoding,
            sep=sep,
            header=header,
            index_col=index_col,
            usecols=usecols,
            dtype=dtype,
            parse_dates=parse_dates,
            na_values=na_values,
            nrows=nrows,
            skiprows=skiprows,
            low_memory=low_memory,
            **kwargs
        )
        
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        return df
        
    except UnicodeDecodeError:
        # Try with different encodings
        for fallback_encoding in ["latin-1", "cp1252", "iso-8859-1"]:
            try:
                logger.warning(f"Retrying with {fallback_encoding} encoding")
                df = pd.read_csv(
                    filepath,
                    encoding=fallback_encoding,
                    sep=sep,
                    header=header,
                    index_col=index_col,
                    usecols=usecols,
                    dtype=dtype,
                    parse_dates=parse_dates,
                    na_values=na_values,
                    nrows=nrows,
                    skiprows=skiprows,
                    low_memory=low_memory,
                    **kwargs
                )
                logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
                return df
            except UnicodeDecodeError:
                continue
        
        raise


# =============================================================================
# Excel Loading
# =============================================================================

def load_excel(
    filepath: Union[str, Path],
    sheet_name: Union[str, int, List, None] = 0,
    header: Union[int, List[int], None] = 0,
    index_col: Optional[Union[int, str, List]] = None,
    usecols: Optional[Union[str, List]] = None,
    dtype: Optional[Dict] = None,
    parse_dates: Optional[Union[bool, List]] = None,
    na_values: Optional[List] = None,
    nrows: Optional[int] = None,
    skiprows: Optional[Union[int, List]] = None,
    **kwargs
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Load data from an Excel file (.xlsx, .xls).
    
    Args:
        filepath: Path to the Excel file.
        sheet_name: Sheet name or index, or list for multiple sheets.
                    Use None to read all sheets.
        header: Row number(s) to use as column names.
        index_col: Column(s) to use as index.
        usecols: Columns to load.
        dtype: Data types for columns.
        parse_dates: Columns to parse as dates.
        na_values: Additional values to treat as NA.
        nrows: Number of rows to read.
        skiprows: Rows to skip.
        **kwargs: Additional arguments passed to pd.read_excel.
    
    Returns:
        pd.DataFrame or Dict[str, pd.DataFrame]: Loaded data.
    
    Example:
        >>> df = load_excel("data.xlsx")
        >>> df = load_excel("data.xlsx", sheet_name="Sheet1")
        >>> sheets = load_excel("data.xlsx", sheet_name=None)  # All sheets
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    logger.info(f"Loading Excel: {filepath}")
    
    result = pd.read_excel(
        filepath,
        sheet_name=sheet_name,
        header=header,
        index_col=index_col,
        usecols=usecols,
        dtype=dtype,
        parse_dates=parse_dates,
        na_values=na_values,
        nrows=nrows,
        skiprows=skiprows,
        **kwargs
    )
    
    if isinstance(result, dict):
        for name, df in result.items():
            logger.info(f"Sheet '{name}': {len(df)} rows, {len(df.columns)} columns")
    else:
        logger.info(f"Loaded {len(result)} rows, {len(result.columns)} columns")
    
    return result


def get_excel_sheet_names(filepath: Union[str, Path]) -> List[str]:
    """
    Get sheet names from an Excel file.
    
    Args:
        filepath: Path to the Excel file.
    
    Returns:
        List[str]: List of sheet names.
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    excel_file = pd.ExcelFile(filepath)
    return excel_file.sheet_names


# =============================================================================
# JSON Loading
# =============================================================================

def load_json(
    filepath: Union[str, Path],
    orient: Optional[str] = None,
    lines: bool = False,
    encoding: str = "utf-8",
    **kwargs
) -> pd.DataFrame:
    """
    Load data from a JSON file.
    
    Args:
        filepath: Path to the JSON file.
        orient: JSON structure orientation. Options:
                'split', 'records', 'index', 'columns', 'values', 'table'
        lines: Read file as line-delimited JSON.
        encoding: File encoding.
        **kwargs: Additional arguments passed to pd.read_json.
    
    Returns:
        pd.DataFrame: Loaded data.
    
    Example:
        >>> df = load_json("data.json")
        >>> df = load_json("data.jsonl", lines=True)
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    logger.info(f"Loading JSON: {filepath}")
    
    df = pd.read_json(
        filepath,
        orient=orient,
        lines=lines,
        encoding=encoding,
        **kwargs
    )
    
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def load_json_nested(
    filepath: Union[str, Path],
    record_path: Optional[Union[str, List[str]]] = None,
    meta: Optional[List[str]] = None,
    max_level: Optional[int] = None,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    Load nested JSON data and normalize it to a flat DataFrame.
    
    Args:
        filepath: Path to the JSON file.
        record_path: Path to the records in the JSON structure.
        meta: Fields to include from higher level.
        max_level: Maximum normalization depth.
        encoding: File encoding.
    
    Returns:
        pd.DataFrame: Normalized data.
    
    Example:
        >>> # For JSON like: {"data": [{"id": 1, "info": {"name": "A"}}]}
        >>> df = load_json_nested("data.json", record_path="data")
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    logger.info(f"Loading nested JSON: {filepath}")
    
    with open(filepath, "r", encoding=encoding) as f:
        data = json.load(f)
    
    df = pd.json_normalize(
        data,
        record_path=record_path,
        meta=meta,
        max_level=max_level,
    )
    
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


# =============================================================================
# SQL Loading
# =============================================================================

def load_sql(
    query: str,
    connection_string: str,
    params: Optional[Dict] = None,
    index_col: Optional[Union[str, List[str]]] = None,
    parse_dates: Optional[Union[List[str], Dict]] = None,
    chunksize: Optional[int] = None,
    **kwargs
) -> Union[pd.DataFrame, pd.io.sql.SQLiteDatabase]:
    """
    Load data from a SQL database.
    
    Supports SQLite, PostgreSQL, MySQL, and other SQLAlchemy-compatible databases.
    
    Args:
        query: SQL query to execute.
        connection_string: Database connection string.
                           Examples:
                           - SQLite: "sqlite:///database.db"
                           - PostgreSQL: "postgresql://user:pass@host:port/db"
                           - MySQL: "mysql+pymysql://user:pass@host:port/db"
        params: Query parameters.
        index_col: Column(s) to use as index.
        parse_dates: Columns to parse as dates.
        chunksize: Number of rows per chunk (for large datasets).
        **kwargs: Additional arguments passed to pd.read_sql.
    
    Returns:
        pd.DataFrame or Iterator: Loaded data.
    
    Example:
        >>> df = load_sql("SELECT * FROM users", "sqlite:///app.db")
        >>> df = load_sql(
        ...     "SELECT * FROM orders WHERE date > :date",
        ...     "postgresql://user:pass@localhost:5432/shop",
        ...     params={"date": "2023-01-01"}
        ... )
    """
    if not SQLALCHEMY_AVAILABLE:
        raise ImportError("SQLAlchemy is required for SQL loading. Install with: pip install sqlalchemy")
    
    logger.info(f"Loading from SQL database")
    
    engine = create_engine(connection_string)
    
    # Use text() for raw SQL queries with params
    if params:
        query = text(query)
    
    df = pd.read_sql(
        query,
        engine,
        params=params,
        index_col=index_col,
        parse_dates=parse_dates,
        chunksize=chunksize,
        **kwargs
    )
    
    if chunksize is None:
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    else:
        logger.info(f"Created chunked reader with chunksize={chunksize}")
    
    return df


def load_sql_table(
    table_name: str,
    connection_string: str,
    schema: Optional[str] = None,
    columns: Optional[List[str]] = None,
    index_col: Optional[Union[str, List[str]]] = None,
    chunksize: Optional[int] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Load an entire table from a SQL database.
    
    Args:
        table_name: Name of the table to load.
        connection_string: Database connection string.
        schema: Database schema.
        columns: Columns to load (None for all).
        index_col: Column(s) to use as index.
        chunksize: Number of rows per chunk.
        **kwargs: Additional arguments.
    
    Returns:
        pd.DataFrame: Loaded data.
    """
    if not SQLALCHEMY_AVAILABLE:
        raise ImportError("SQLAlchemy is required for SQL loading. Install with: pip install sqlalchemy")
    
    logger.info(f"Loading table: {table_name}")
    
    engine = create_engine(connection_string)
    
    df = pd.read_sql_table(
        table_name,
        engine,
        schema=schema,
        columns=columns,
        index_col=index_col,
        chunksize=chunksize,
        **kwargs
    )
    
    if chunksize is None:
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    return df


# =============================================================================
# API/URL Loading
# =============================================================================

def load_url(
    url: str,
    format: str = "csv",
    params: Optional[Dict] = None,
    headers: Optional[Dict] = None,
    auth: Optional[tuple] = None,
    timeout: int = 30,
    **kwargs
) -> pd.DataFrame:
    """
    Load data from a URL.
    
    Args:
        url: URL to load data from.
        format: Data format ('csv', 'json', 'excel').
        params: Query parameters.
        headers: HTTP headers.
        auth: Authentication tuple (username, password).
        timeout: Request timeout in seconds.
        **kwargs: Additional arguments for the format loader.
    
    Returns:
        pd.DataFrame: Loaded data.
    
    Example:
        >>> df = load_url("https://example.com/data.csv")
        >>> df = load_url(
        ...     "https://api.example.com/data",
        ...     format="json",
        ...     headers={"Authorization": "Bearer token"}
        ... )
    """
    if not REQUESTS_AVAILABLE:
        raise ImportError("requests is required for URL loading. Install with: pip install requests")
    
    logger.info(f"Loading from URL: {url}")
    
    response = requests.get(
        url,
        params=params,
        headers=headers,
        auth=auth,
        timeout=timeout,
    )
    response.raise_for_status()
    
    content = io.BytesIO(response.content)
    
    if format == "csv":
        df = pd.read_csv(content, **kwargs)
    elif format == "json":
        df = pd.read_json(content, **kwargs)
    elif format == "excel":
        df = pd.read_excel(content, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def load_api(
    url: str,
    method: str = "GET",
    params: Optional[Dict] = None,
    data: Optional[Dict] = None,
    json_data: Optional[Dict] = None,
    headers: Optional[Dict] = None,
    auth: Optional[tuple] = None,
    timeout: int = 30,
    data_key: Optional[str] = None,
    paginate: bool = False,
    page_key: str = "page",
    limit_key: str = "limit",
    limit: int = 100,
    max_pages: int = 100,
) -> pd.DataFrame:
    """
    Load data from a REST API with pagination support.
    
    Args:
        url: API endpoint URL.
        method: HTTP method.
        params: Query parameters.
        data: Form data.
        json_data: JSON body data.
        headers: HTTP headers.
        auth: Authentication tuple.
        timeout: Request timeout.
        data_key: Key in response containing the data array.
        paginate: Whether to paginate through results.
        page_key: Parameter name for page number.
        limit_key: Parameter name for page size.
        limit: Number of items per page.
        max_pages: Maximum number of pages to fetch.
    
    Returns:
        pd.DataFrame: Loaded data.
    
    Example:
        >>> df = load_api(
        ...     "https://api.example.com/users",
        ...     headers={"Authorization": "Bearer token"},
        ...     data_key="users",
        ...     paginate=True
        ... )
    """
    if not REQUESTS_AVAILABLE:
        raise ImportError("requests is required for API loading. Install with: pip install requests")
    
    logger.info(f"Loading from API: {url}")
    
    all_data = []
    page = 1
    
    while True:
        # Build params for this request
        request_params = dict(params or {})
        if paginate:
            request_params[page_key] = page
            request_params[limit_key] = limit
        
        response = requests.request(
            method=method,
            url=url,
            params=request_params,
            data=data,
            json=json_data,
            headers=headers,
            auth=auth,
            timeout=timeout,
        )
        response.raise_for_status()
        
        result = response.json()
        
        # Extract data
        if data_key:
            page_data = result.get(data_key, [])
        else:
            page_data = result if isinstance(result, list) else [result]
        
        all_data.extend(page_data)
        
        # Check if we should continue paginating
        if not paginate or len(page_data) < limit or page >= max_pages:
            break
        
        page += 1
        logger.debug(f"Fetching page {page}...")
    
    df = pd.DataFrame(all_data)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


# =============================================================================
# Compressed Files
# =============================================================================

def load_compressed(
    filepath: Union[str, Path],
    format: str = "csv",
    compression: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Load data from a compressed file (.zip, .gz, .bz2, .xz).
    
    Args:
        filepath: Path to the compressed file.
        format: Data format inside the archive ('csv', 'json', 'excel').
        compression: Compression type. Auto-detected if None.
        **kwargs: Additional arguments for the format loader.
    
    Returns:
        pd.DataFrame: Loaded data.
    
    Example:
        >>> df = load_compressed("data.csv.gz")
        >>> df = load_compressed("archive.zip", format="csv")
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Auto-detect compression type
    if compression is None:
        suffix = filepath.suffix.lower()
        if suffix == ".gz":
            compression = "gzip"
        elif suffix == ".bz2":
            compression = "bz2"
        elif suffix == ".xz":
            compression = "xz"
        elif suffix == ".zip":
            compression = "zip"
        else:
            compression = "infer"
    
    logger.info(f"Loading compressed file: {filepath} ({compression})")
    
    if compression == "zip":
        return _load_from_zip(filepath, format, **kwargs)
    else:
        if format == "csv":
            df = pd.read_csv(filepath, compression=compression, **kwargs)
        elif format == "json":
            df = pd.read_json(filepath, compression=compression, **kwargs)
        else:
            raise ValueError(f"Unsupported format for compression: {format}")
    
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def _load_from_zip(
    filepath: Union[str, Path],
    format: str = "csv",
    file_pattern: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """Load data from within a ZIP archive."""
    with zipfile.ZipFile(filepath, "r") as z:
        file_list = z.namelist()
        
        # Filter files by pattern or extension
        if file_pattern:
            import fnmatch
            matching_files = [f for f in file_list if fnmatch.fnmatch(f, file_pattern)]
        else:
            ext = f".{format}"
            matching_files = [f for f in file_list if f.endswith(ext)]
        
        if not matching_files:
            raise ValueError(f"No {format} files found in archive")
        
        # Load the first matching file (or concatenate all)
        if len(matching_files) == 1:
            with z.open(matching_files[0]) as f:
                content = io.BytesIO(f.read())
                if format == "csv":
                    return pd.read_csv(content, **kwargs)
                elif format == "json":
                    return pd.read_json(content, **kwargs)
                elif format == "excel":
                    return pd.read_excel(content, **kwargs)
        else:
            # Concatenate all matching files
            dfs = []
            for filename in matching_files:
                with z.open(filename) as f:
                    content = io.BytesIO(f.read())
                    if format == "csv":
                        df = pd.read_csv(content, **kwargs)
                    elif format == "json":
                        df = pd.read_json(content, **kwargs)
                    dfs.append(df)
            return pd.concat(dfs, ignore_index=True)


# =============================================================================
# Auto Loader
# =============================================================================

def load_auto(
    source: Union[str, Path],
    **kwargs
) -> pd.DataFrame:
    """
    Automatically detect and load data from various sources.
    
    Supports CSV, Excel, JSON, SQL, and compressed files.
    Automatically detects the format based on file extension or URL.
    
    Args:
        source: Path to file, URL, or SQL connection string.
        **kwargs: Additional arguments passed to the appropriate loader.
    
    Returns:
        pd.DataFrame: Loaded data.
    
    Example:
        >>> df = load_auto("data.csv")
        >>> df = load_auto("https://example.com/data.json")
        >>> df = load_auto("data.xlsx")
    """
    source_str = str(source)
    
    # Check if it's a URL
    if source_str.startswith(("http://", "https://")):
        parsed = urlparse(source_str)
        path = parsed.path.lower()
        
        if path.endswith(".csv"):
            return load_url(source_str, format="csv", **kwargs)
        elif path.endswith(".json") or path.endswith(".jsonl"):
            return load_url(source_str, format="json", **kwargs)
        elif path.endswith((".xlsx", ".xls")):
            return load_url(source_str, format="excel", **kwargs)
        else:
            # Try JSON by default for API endpoints
            return load_url(source_str, format="json", **kwargs)
    
    # It's a file path
    filepath = Path(source)
    suffix = filepath.suffix.lower()
    
    # Remove compression suffix to get actual format
    if suffix in [".gz", ".bz2", ".xz", ".zip"]:
        if suffix == ".zip":
            return load_compressed(filepath, **kwargs)
        
        # Get the format from the second-to-last suffix
        stem = filepath.stem
        inner_suffix = Path(stem).suffix.lower()
        
        if inner_suffix == ".csv":
            return load_compressed(filepath, format="csv", **kwargs)
        elif inner_suffix == ".json":
            return load_compressed(filepath, format="json", **kwargs)
        else:
            return load_compressed(filepath, format="csv", **kwargs)
    
    # Standard file types
    if suffix == ".csv":
        return load_csv(filepath, **kwargs)
    elif suffix in [".xlsx", ".xls"]:
        return load_excel(filepath, **kwargs)
    elif suffix in [".json", ".jsonl"]:
        lines = suffix == ".jsonl"
        return load_json(filepath, lines=lines, **kwargs)
    elif suffix == ".parquet":
        return pd.read_parquet(filepath, **kwargs)
    elif suffix == ".feather":
        return pd.read_feather(filepath, **kwargs)
    elif suffix == ".pickle" or suffix == ".pkl":
        return pd.read_pickle(filepath, **kwargs)
    else:
        # Try CSV as default
        logger.warning(f"Unknown file format: {suffix}, trying CSV")
        return load_csv(filepath, **kwargs)


# =============================================================================
# Data Saving
# =============================================================================

def save_csv(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    index: bool = False,
    encoding: str = "utf-8",
    **kwargs
) -> None:
    """
    Save DataFrame to CSV file.
    
    Args:
        df: DataFrame to save.
        filepath: Output file path.
        index: Whether to include index.
        encoding: File encoding.
        **kwargs: Additional arguments passed to df.to_csv.
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    df.to_csv(filepath, index=index, encoding=encoding, **kwargs)
    logger.info(f"Saved {len(df)} rows to {filepath}")


def save_excel(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    sheet_name: str = "Sheet1",
    index: bool = False,
    **kwargs
) -> None:
    """
    Save DataFrame to Excel file.
    
    Args:
        df: DataFrame to save.
        filepath: Output file path.
        sheet_name: Name of the sheet.
        index: Whether to include index.
        **kwargs: Additional arguments.
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    df.to_excel(filepath, sheet_name=sheet_name, index=index, **kwargs)
    logger.info(f"Saved {len(df)} rows to {filepath}")


def save_json(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    orient: str = "records",
    indent: int = 2,
    **kwargs
) -> None:
    """
    Save DataFrame to JSON file.
    
    Args:
        df: DataFrame to save.
        filepath: Output file path.
        orient: JSON structure orientation.
        indent: Indentation level.
        **kwargs: Additional arguments.
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    df.to_json(filepath, orient=orient, indent=indent, **kwargs)
    logger.info(f"Saved {len(df)} rows to {filepath}")
