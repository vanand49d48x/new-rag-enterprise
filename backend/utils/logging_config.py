import logging
import sys
from datetime import datetime
import time

def setup_logging():
    """Setup centralized logging configuration with timestamps in local timezone"""
    # Create a custom formatter that uses local timezone
    class LocalTimeFormatter(logging.Formatter):
        def formatTime(self, record, datefmt=None):
            # Convert to local timezone
            local_time = datetime.fromtimestamp(record.created)
            if datefmt:
                return local_time.strftime(datefmt)
            else:
                return local_time.strftime('%Y-%m-%d %H:%M:%S')
    
    # Create formatter with local timezone
    formatter = LocalTimeFormatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup handlers with local timezone formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    file_handler = logging.FileHandler('logs/rag_system.log')
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler, file_handler],
        force=True  # Override any existing configuration
    )

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name"""
    return logging.getLogger(name)

def log_with_timestamp(logger: logging.Logger, level: str, message: str):
    """Log a message with timestamp in local timezone"""
    # Use local timezone
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    formatted_message = f"[{timestamp}] {message}"
    
    if level.upper() == 'INFO':
        logger.info(formatted_message)
    elif level.upper() == 'ERROR':
        logger.error(formatted_message)
    elif level.upper() == 'WARNING':
        logger.warning(formatted_message)
    elif level.upper() == 'DEBUG':
        logger.debug(formatted_message)
    else:
        logger.info(formatted_message) 