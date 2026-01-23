# API Reference

## adamops.data.loaders

### load_csv(filepath, encoding=None, auto_detect_encoding=True, ...)
Load data from a CSV file with auto-encoding detection.

**Parameters:**
- `filepath`: Path to the CSV file
- `encoding`: File encoding (auto-detected if None)
- `sep`: Column separator (default: ',')
- `parse_dates`: Columns to parse as dates

**Returns:** pandas DataFrame

### load_excel(filepath, sheet_name=0, ...)
Load data from an Excel file.

### load_json(filepath, orient=None, lines=False, ...)
Load data from a JSON file.

### load_sql(query, connection_string, ...)
Load data from a SQL database.

### load_auto(source, ...)
Automatically detect and load data from various sources.

---

## adamops.data.validators

### validate(df, schema=None, required_columns=None, ...)
Validate a DataFrame and generate a ValidationReport.

### check_missing(df, threshold=0.0)
Check missing values in a DataFrame.

### check_duplicates(df, subset=None)
Get duplicate rows from a DataFrame.

---

## adamops.models.modelops

### train(X, y, task='auto', algorithm='random_forest', params=None)
Train a model.

**Parameters:**
- `X`: Features
- `y`: Target
- `task`: 'classification', 'regression', or 'auto'
- `algorithm`: Model algorithm
- `params`: Hyperparameters

**Returns:** TrainedModel

### compare_models(X, y, task, algorithms=None, cv=5)
Compare multiple models using cross-validation.

---

## adamops.models.automl

### run(X, y, task='auto', tuning='bayesian', time_limit=3600, ...)
Run AutoML.

**Returns:** AutoMLResult with best model and leaderboard

---

## adamops.evaluation.metrics

### evaluate(y_true, y_pred, task='auto', y_prob=None)
Unified evaluation function.

### classification_metrics(y_true, y_pred, y_prob=None)
Compute classification metrics.

### regression_metrics(y_true, y_pred)
Compute regression metrics.

---

## adamops.deployment.api

### create_fastapi_app(model, model_name='model')
Create FastAPI application for model serving.

### run_api(model, framework='fastapi', host='0.0.0.0', port=8000)
Run model serving API.
