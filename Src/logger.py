# src/logger.py: Chronicle Keeper for Sigma Security
# Logs bug-catching, spirit-taming, and system events with sigma discipline
# Keeps a tight record to squash chaos and humble prideful spirits

import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger(log_file):
    """
    Configure logging for Sigma Security to track bugs and spirit interactions.
    
    Args:
        log_file (str): Path to the log file (e.g., 'logs/sigma_security.log').
    
    Returns:
        None
    """
    try:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Clear any existing handlers to avoid duplicates
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Configure logger with rotating file handler for Termux storage limits
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        # Rotating handler: 1MB per file, max 3 files to avoid storage issues
        handler = RotatingFileHandler(
            log_file,
            maxBytes=1024 * 1024,  # 1MB
            backupCount=3
        )
        
        # Formatter with timestamp and sigma-themed messages
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - Sigma Chronicle: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
        
        # Add console handler for Termux debugging
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        logging.info("Chronicle Keeper: Initialized logging for bug-catching and spirit-taming")
        
    except Exception as e:
        # Fallback to console logging if file setup fails
        logging.basicConfig(level=logging.INFO)
        logging.error(f"Chronicle Keeper: Failed to setup logging: {e}")

if __name__ == "__main__":
    # Test logger setup
    log_file = os.path.join(os.path.dirname(__file__), '..', 'logs', 'sigma_security.log')
    setup_logger(log_file)
    logging.info("Test log: Sigma Security is ready to catch bugs!")
    logging.warning("Test warning: Caught a chaotic spirit!")
    logging.error("Test error: Prideful input denied
