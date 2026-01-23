"""
AdamOps Logging Module

Provides centralized logging functionality for the entire library.
Supports console and file logging with configurable levels and formats.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
from logging.handlers import RotatingFileHandler
from datetime import datetime

# Custom log levels
TRACE = 5
logging.addLevelName(TRACE, "TRACE")

# Module-level logger cache
_loggers: dict = {}

# Default format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class ColoredFormatter(logging.Formatter):
    """
    Colored formatter for console output.
    
    Adds color codes to log messages based on log level.
    """
    
    # ANSI color codes
    COLORS = {
        "TRACE": "\033[37m",      # White
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[35m",   # Magenta
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None, use_colors: bool = True):
        super().__init__(fmt or DEFAULT_FORMAT, datefmt or DEFAULT_DATE_FORMAT)
        self.use_colors = use_colors
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with optional colors."""
        # Save original levelname
        original_levelname = record.levelname
        
        if self.use_colors and record.levelname in self.COLORS:
            color = self.COLORS[record.levelname]
            record.levelname = f"{color}{self.BOLD}{record.levelname}{self.RESET}"
            record.msg = f"{color}{record.msg}{self.RESET}"
        
        result = super().format(record)
        
        # Restore original levelname
        record.levelname = original_levelname
        
        return result


class AdamOpsLogger:
    """
    Custom logger class for AdamOps.
    
    Provides a unified logging interface with features like:
    - Console and file logging
    - Colored output
    - Automatic log rotation
    - Context managers for temporary log level changes
    
    Example:
        >>> logger = AdamOpsLogger("my_module")
        >>> logger.info("This is an info message")
        >>> logger.debug("This is a debug message")
    """
    
    def __init__(
        self,
        name: str,
        level: str = "INFO",
        log_file: Optional[str] = None,
        console: bool = True,
        use_colors: bool = True,
        format_string: Optional[str] = None,
        max_bytes: int = 10485760,  # 10MB
        backup_count: int = 5,
    ):
        """
        Initialize the logger.
        
        Args:
            name: Logger name (usually module name).
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            log_file: Optional path to log file.
            console: Whether to log to console.
            use_colors: Whether to use colored output.
            format_string: Custom format string.
            max_bytes: Maximum log file size before rotation.
            backup_count: Number of backup files to keep.
        """
        self.name = name
        self._logger = logging.getLogger(name)
        self._logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        self._logger.handlers = []  # Clear existing handlers
        
        format_str = format_string or DEFAULT_FORMAT
        
        # Console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(ColoredFormatter(format_str, use_colors=use_colors))
            self._logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(logging.Formatter(format_str, DEFAULT_DATE_FORMAT))
            self._logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        self._logger.propagate = False
    
    def set_level(self, level: str) -> None:
        """Set the log level."""
        self._logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    def trace(self, msg: str, *args, **kwargs) -> None:
        """Log a TRACE level message."""
        self._logger.log(TRACE, msg, *args, **kwargs)
    
    def debug(self, msg: str, *args, **kwargs) -> None:
        """Log a DEBUG level message."""
        self._logger.debug(msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs) -> None:
        """Log an INFO level message."""
        self._logger.info(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs) -> None:
        """Log a WARNING level message."""
        self._logger.warning(msg, *args, **kwargs)
    
    def warn(self, msg: str, *args, **kwargs) -> None:
        """Alias for warning."""
        self.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs) -> None:
        """Log an ERROR level message."""
        self._logger.error(msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs) -> None:
        """Log a CRITICAL level message."""
        self._logger.critical(msg, *args, **kwargs)
    
    def exception(self, msg: str, *args, **kwargs) -> None:
        """Log an exception with traceback."""
        self._logger.exception(msg, *args, **kwargs)
    
    def log(self, level: int, msg: str, *args, **kwargs) -> None:
        """Log a message at the specified level."""
        self._logger.log(level, msg, *args, **kwargs)


def get_logger(
    name: str = "adamops",
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    console: bool = True,
    use_colors: bool = True,
) -> AdamOpsLogger:
    """
    Get or create a logger with the specified name.
    
    Args:
        name: Logger name.
        level: Log level (default: INFO).
        log_file: Optional path to log file.
        console: Whether to log to console.
        use_colors: Whether to use colored output.
    
    Returns:
        AdamOpsLogger: Logger instance.
    
    Example:
        >>> logger = get_logger("my_module")
        >>> logger.info("Processing data...")
    """
    # Use default level from config if not specified
    if level is None:
        try:
            from adamops.utils.config import get_config
            config = get_config()
            level = config.logging.level
            if log_file is None:
                log_file = config.logging.file
        except ImportError:
            level = "INFO"
    
    # Check cache
    cache_key = f"{name}:{level}:{log_file}:{console}:{use_colors}"
    if cache_key not in _loggers:
        _loggers[cache_key] = AdamOpsLogger(
            name=name,
            level=level,
            log_file=log_file,
            console=console,
            use_colors=use_colors,
        )
    
    return _loggers[cache_key]


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True,
    use_colors: bool = True,
    format_string: Optional[str] = None,
) -> None:
    """
    Set up global logging configuration for AdamOps.
    
    Args:
        level: Global log level.
        log_file: Path to log file.
        console: Whether to log to console.
        use_colors: Whether to use colored output.
        format_string: Custom format string.
    
    Example:
        >>> setup_logging(level="DEBUG", log_file="adamops.log")
    """
    # Configure root adamops logger
    root_logger = logging.getLogger("adamops")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    root_logger.handlers = []
    
    format_str = format_string or DEFAULT_FORMAT
    
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColoredFormatter(format_str, use_colors=use_colors))
        root_logger.addHandler(console_handler)
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10485760,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(format_str, DEFAULT_DATE_FORMAT))
        root_logger.addHandler(file_handler)


class LogContext:
    """
    Context manager for temporary log level changes.
    
    Example:
        >>> logger = get_logger("my_module")
        >>> with LogContext(logger, "DEBUG"):
        ...     logger.debug("This will be logged")
        >>> logger.debug("This might not be logged")
    """
    
    def __init__(self, logger: Union[AdamOpsLogger, logging.Logger], level: str):
        """
        Initialize context manager.
        
        Args:
            logger: Logger to modify.
            level: Temporary log level.
        """
        self.logger = logger
        self.new_level = getattr(logging, level.upper(), logging.INFO)
        self.old_level = None
    
    def __enter__(self) -> None:
        """Enter context and set new level."""
        if isinstance(self.logger, AdamOpsLogger):
            self.old_level = self.logger._logger.level
            self.logger._logger.setLevel(self.new_level)
        else:
            self.old_level = self.logger.level
            self.logger.setLevel(self.new_level)
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and restore old level."""
        if isinstance(self.logger, AdamOpsLogger):
            self.logger._logger.setLevel(self.old_level)
        else:
            self.logger.setLevel(self.old_level)


class Timer:
    """
    Context manager for timing operations with optional logging.
    
    Example:
        >>> logger = get_logger("my_module")
        >>> with Timer("Data loading", logger):
        ...     load_data()
        [INFO] Data loading completed in 2.34s
    """
    
    def __init__(self, operation: str, logger: Optional[AdamOpsLogger] = None, level: str = "INFO"):
        """
        Initialize timer.
        
        Args:
            operation: Name of the operation being timed.
            logger: Optional logger for timing output.
            level: Log level for timing message.
        """
        self.operation = operation
        self.logger = logger
        self.level = level.upper()
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()
    
    def __enter__(self) -> "Timer":
        """Start timer."""
        self.start_time = datetime.now()
        if self.logger:
            self.logger.log(
                getattr(logging, self.level, logging.INFO),
                f"Starting: {self.operation}..."
            )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop timer and log result."""
        self.end_time = datetime.now()
        if self.logger:
            status = "completed" if exc_type is None else "failed"
            self.logger.log(
                getattr(logging, self.level, logging.INFO),
                f"{self.operation} {status} in {self.elapsed:.2f}s"
            )


def log_function_call(logger: Optional[AdamOpsLogger] = None, level: str = "DEBUG"):
    """
    Decorator to log function calls and their results.
    
    Args:
        logger: Logger to use (creates one if not provided).
        level: Log level for function call messages.
    
    Example:
        >>> @log_function_call()
        ... def process_data(df):
        ...     return df.dropna()
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)
            
            log_level = getattr(logging, level.upper(), logging.DEBUG)
            
            # Log function call
            args_str = ", ".join([repr(a) for a in args[:3]])  # Limit args
            if len(args) > 3:
                args_str += ", ..."
            kwargs_str = ", ".join([f"{k}={repr(v)}" for k, v in list(kwargs.items())[:3]])
            if len(kwargs) > 3:
                kwargs_str += ", ..."
            
            call_str = f"{func.__name__}({args_str}{', ' if args_str and kwargs_str else ''}{kwargs_str})"
            logger.log(log_level, f"Calling: {call_str}")
            
            try:
                result = func(*args, **kwargs)
                logger.log(log_level, f"{func.__name__} returned successfully")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} raised {type(e).__name__}: {e}")
                raise
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator
