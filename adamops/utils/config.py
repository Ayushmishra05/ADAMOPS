"""
AdamOps Configuration Module

Provides centralized configuration management for the entire library.
Supports YAML, JSON, and environment variable configurations.
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


@dataclass
class DataConfig:
    """Configuration for data module."""
    default_encoding: str = "utf-8"
    missing_threshold: float = 0.5
    outlier_method: str = "iqr"
    outlier_threshold: float = 1.5
    validation_sample_size: int = 10000
    auto_detect_types: bool = True


@dataclass
class ModelConfig:
    """Configuration for model module."""
    default_random_state: int = 42
    cv_folds: int = 5
    early_stopping_rounds: int = 50
    n_jobs: int = -1
    verbose: int = 0


@dataclass
class AutoMLConfig:
    """Configuration for AutoML module."""
    time_limit: int = 3600
    max_trials: int = 100
    tuning_method: str = "bayesian"
    optimization_metric: str = "auto"
    early_stopping: bool = True


@dataclass
class DeploymentConfig:
    """Configuration for deployment module."""
    default_port: int = 8000
    default_host: str = "0.0.0.0"
    api_framework: str = "fastapi"
    enable_cors: bool = True
    log_requests: bool = True


@dataclass
class MonitoringConfig:
    """Configuration for monitoring module."""
    drift_threshold: float = 0.05
    alert_email: Optional[str] = None
    check_interval: int = 3600
    log_predictions: bool = True


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None
    console: bool = True
    max_bytes: int = 10485760  # 10MB
    backup_count: int = 5


@dataclass
class AdamOpsConfig:
    """Main configuration class for AdamOps."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    automl: AutoMLConfig = field(default_factory=AutoMLConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Registry settings
    registry_backend: str = "json"  # json or sqlite
    registry_path: str = ".adamops_registry"
    
    # General settings
    cache_enabled: bool = True
    cache_path: str = ".adamops_cache"


# Global configuration instance
_config: Optional[AdamOpsConfig] = None


def get_config() -> AdamOpsConfig:
    """
    Get the global configuration instance.
    
    Returns:
        AdamOpsConfig: The global configuration object.
    
    Example:
        >>> config = get_config()
        >>> print(config.model.cv_folds)
        5
    """
    global _config
    if _config is None:
        _config = AdamOpsConfig()
    return _config


def set_config(config: AdamOpsConfig) -> None:
    """
    Set the global configuration instance.
    
    Args:
        config: The configuration object to set as global.
    
    Example:
        >>> custom_config = AdamOpsConfig()
        >>> custom_config.model.cv_folds = 10
        >>> set_config(custom_config)
    """
    global _config
    _config = config


def reset_config() -> None:
    """
    Reset the global configuration to defaults.
    
    Example:
        >>> reset_config()
        >>> config = get_config()
        >>> print(config.model.cv_folds)
        5
    """
    global _config
    _config = AdamOpsConfig()


def load_config_from_file(filepath: Union[str, Path]) -> AdamOpsConfig:
    """
    Load configuration from a YAML or JSON file.
    
    Args:
        filepath: Path to the configuration file.
    
    Returns:
        AdamOpsConfig: Loaded configuration object.
    
    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file format is not supported.
    
    Example:
        >>> config = load_config_from_file("config.yaml")
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        if filepath.suffix in [".yaml", ".yml"]:
            if not YAML_AVAILABLE:
                raise ImportError("PyYAML is required to load YAML config files. Install with: pip install pyyaml")
            config_dict = yaml.safe_load(f)
        elif filepath.suffix == ".json":
            config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {filepath.suffix}")
    
    return _dict_to_config(config_dict)


def save_config_to_file(config: AdamOpsConfig, filepath: Union[str, Path]) -> None:
    """
    Save configuration to a YAML or JSON file.
    
    Args:
        config: Configuration object to save.
        filepath: Path to save the configuration to.
    
    Example:
        >>> config = get_config()
        >>> save_config_to_file(config, "config.yaml")
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = _config_to_dict(config)
    
    with open(filepath, "w", encoding="utf-8") as f:
        if filepath.suffix in [".yaml", ".yml"]:
            if not YAML_AVAILABLE:
                raise ImportError("PyYAML is required to save YAML config files. Install with: pip install pyyaml")
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif filepath.suffix == ".json":
            json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {filepath.suffix}")


def load_config_from_env(prefix: str = "ADAMOPS") -> AdamOpsConfig:
    """
    Load configuration from environment variables.
    
    Environment variables should be named as {prefix}_{SECTION}_{KEY}.
    For example: ADAMOPS_MODEL_CV_FOLDS=10
    
    Args:
        prefix: Prefix for environment variables.
    
    Returns:
        AdamOpsConfig: Configuration with values from environment.
    
    Example:
        >>> # Set env: ADAMOPS_MODEL_CV_FOLDS=10
        >>> config = load_config_from_env()
        >>> print(config.model.cv_folds)
        10
    """
    if DOTENV_AVAILABLE:
        load_dotenv()
    
    config = AdamOpsConfig()
    
    # Map of environment variable suffixes to config attributes
    env_mappings = {
        # Data config
        f"{prefix}_DATA_DEFAULT_ENCODING": ("data", "default_encoding", str),
        f"{prefix}_DATA_MISSING_THRESHOLD": ("data", "missing_threshold", float),
        f"{prefix}_DATA_OUTLIER_METHOD": ("data", "outlier_method", str),
        f"{prefix}_DATA_OUTLIER_THRESHOLD": ("data", "outlier_threshold", float),
        
        # Model config
        f"{prefix}_MODEL_RANDOM_STATE": ("model", "default_random_state", int),
        f"{prefix}_MODEL_CV_FOLDS": ("model", "cv_folds", int),
        f"{prefix}_MODEL_N_JOBS": ("model", "n_jobs", int),
        
        # AutoML config
        f"{prefix}_AUTOML_TIME_LIMIT": ("automl", "time_limit", int),
        f"{prefix}_AUTOML_MAX_TRIALS": ("automl", "max_trials", int),
        f"{prefix}_AUTOML_TUNING_METHOD": ("automl", "tuning_method", str),
        
        # Deployment config
        f"{prefix}_DEPLOY_PORT": ("deployment", "default_port", int),
        f"{prefix}_DEPLOY_HOST": ("deployment", "default_host", str),
        f"{prefix}_DEPLOY_FRAMEWORK": ("deployment", "api_framework", str),
        
        # Monitoring config
        f"{prefix}_MONITOR_DRIFT_THRESHOLD": ("monitoring", "drift_threshold", float),
        f"{prefix}_MONITOR_CHECK_INTERVAL": ("monitoring", "check_interval", int),
        
        # Logging config
        f"{prefix}_LOG_LEVEL": ("logging", "level", str),
        f"{prefix}_LOG_FILE": ("logging", "file", str),
        
        # General settings
        f"{prefix}_REGISTRY_BACKEND": (None, "registry_backend", str),
        f"{prefix}_REGISTRY_PATH": (None, "registry_path", str),
    }
    
    for env_var, (section, attr, type_conv) in env_mappings.items():
        value = os.environ.get(env_var)
        if value is not None:
            try:
                converted_value = type_conv(value)
                if section is not None:
                    setattr(getattr(config, section), attr, converted_value)
                else:
                    setattr(config, attr, converted_value)
            except (ValueError, TypeError):
                pass  # Skip invalid values
    
    return config


def _config_to_dict(config: AdamOpsConfig) -> Dict[str, Any]:
    """Convert configuration object to dictionary."""
    return {
        "data": {
            "default_encoding": config.data.default_encoding,
            "missing_threshold": config.data.missing_threshold,
            "outlier_method": config.data.outlier_method,
            "outlier_threshold": config.data.outlier_threshold,
            "validation_sample_size": config.data.validation_sample_size,
            "auto_detect_types": config.data.auto_detect_types,
        },
        "model": {
            "default_random_state": config.model.default_random_state,
            "cv_folds": config.model.cv_folds,
            "early_stopping_rounds": config.model.early_stopping_rounds,
            "n_jobs": config.model.n_jobs,
            "verbose": config.model.verbose,
        },
        "automl": {
            "time_limit": config.automl.time_limit,
            "max_trials": config.automl.max_trials,
            "tuning_method": config.automl.tuning_method,
            "optimization_metric": config.automl.optimization_metric,
            "early_stopping": config.automl.early_stopping,
        },
        "deployment": {
            "default_port": config.deployment.default_port,
            "default_host": config.deployment.default_host,
            "api_framework": config.deployment.api_framework,
            "enable_cors": config.deployment.enable_cors,
            "log_requests": config.deployment.log_requests,
        },
        "monitoring": {
            "drift_threshold": config.monitoring.drift_threshold,
            "alert_email": config.monitoring.alert_email,
            "check_interval": config.monitoring.check_interval,
            "log_predictions": config.monitoring.log_predictions,
        },
        "logging": {
            "level": config.logging.level,
            "format": config.logging.format,
            "file": config.logging.file,
            "console": config.logging.console,
            "max_bytes": config.logging.max_bytes,
            "backup_count": config.logging.backup_count,
        },
        "registry_backend": config.registry_backend,
        "registry_path": config.registry_path,
        "cache_enabled": config.cache_enabled,
        "cache_path": config.cache_path,
    }


def _dict_to_config(config_dict: Dict[str, Any]) -> AdamOpsConfig:
    """Convert dictionary to configuration object."""
    config = AdamOpsConfig()
    
    # Data config
    if "data" in config_dict:
        data = config_dict["data"]
        config.data = DataConfig(
            default_encoding=data.get("default_encoding", config.data.default_encoding),
            missing_threshold=data.get("missing_threshold", config.data.missing_threshold),
            outlier_method=data.get("outlier_method", config.data.outlier_method),
            outlier_threshold=data.get("outlier_threshold", config.data.outlier_threshold),
            validation_sample_size=data.get("validation_sample_size", config.data.validation_sample_size),
            auto_detect_types=data.get("auto_detect_types", config.data.auto_detect_types),
        )
    
    # Model config
    if "model" in config_dict:
        model = config_dict["model"]
        config.model = ModelConfig(
            default_random_state=model.get("default_random_state", config.model.default_random_state),
            cv_folds=model.get("cv_folds", config.model.cv_folds),
            early_stopping_rounds=model.get("early_stopping_rounds", config.model.early_stopping_rounds),
            n_jobs=model.get("n_jobs", config.model.n_jobs),
            verbose=model.get("verbose", config.model.verbose),
        )
    
    # AutoML config
    if "automl" in config_dict:
        automl = config_dict["automl"]
        config.automl = AutoMLConfig(
            time_limit=automl.get("time_limit", config.automl.time_limit),
            max_trials=automl.get("max_trials", config.automl.max_trials),
            tuning_method=automl.get("tuning_method", config.automl.tuning_method),
            optimization_metric=automl.get("optimization_metric", config.automl.optimization_metric),
            early_stopping=automl.get("early_stopping", config.automl.early_stopping),
        )
    
    # Deployment config
    if "deployment" in config_dict:
        deploy = config_dict["deployment"]
        config.deployment = DeploymentConfig(
            default_port=deploy.get("default_port", config.deployment.default_port),
            default_host=deploy.get("default_host", config.deployment.default_host),
            api_framework=deploy.get("api_framework", config.deployment.api_framework),
            enable_cors=deploy.get("enable_cors", config.deployment.enable_cors),
            log_requests=deploy.get("log_requests", config.deployment.log_requests),
        )
    
    # Monitoring config
    if "monitoring" in config_dict:
        monitor = config_dict["monitoring"]
        config.monitoring = MonitoringConfig(
            drift_threshold=monitor.get("drift_threshold", config.monitoring.drift_threshold),
            alert_email=monitor.get("alert_email", config.monitoring.alert_email),
            check_interval=monitor.get("check_interval", config.monitoring.check_interval),
            log_predictions=monitor.get("log_predictions", config.monitoring.log_predictions),
        )
    
    # Logging config
    if "logging" in config_dict:
        log = config_dict["logging"]
        config.logging = LoggingConfig(
            level=log.get("level", config.logging.level),
            format=log.get("format", config.logging.format),
            file=log.get("file", config.logging.file),
            console=log.get("console", config.logging.console),
            max_bytes=log.get("max_bytes", config.logging.max_bytes),
            backup_count=log.get("backup_count", config.logging.backup_count),
        )
    
    # General settings
    config.registry_backend = config_dict.get("registry_backend", config.registry_backend)
    config.registry_path = config_dict.get("registry_path", config.registry_path)
    config.cache_enabled = config_dict.get("cache_enabled", config.cache_enabled)
    config.cache_path = config_dict.get("cache_path", config.cache_path)
    
    return config


def update_config(**kwargs) -> AdamOpsConfig:
    """
    Update specific configuration values.
    
    Args:
        **kwargs: Configuration values in format section__key=value.
    
    Returns:
        AdamOpsConfig: Updated configuration object.
    
    Example:
        >>> config = update_config(model__cv_folds=10, automl__time_limit=7200)
        >>> print(config.model.cv_folds)
        10
    """
    config = get_config()
    
    for key, value in kwargs.items():
        if "__" in key:
            section, attr = key.split("__", 1)
            if hasattr(config, section):
                section_config = getattr(config, section)
                if hasattr(section_config, attr):
                    setattr(section_config, attr, value)
        elif hasattr(config, key):
            setattr(config, key, value)
    
    return config
