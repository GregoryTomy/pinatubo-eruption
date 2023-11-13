import logging

def setup_logger(name, log_file, level=logging.INFO):
    """Function to set up a logger with a file and stream handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:  # Check if logger already has handlers
        logger.setLevel(level)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s")
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    
    return logger
