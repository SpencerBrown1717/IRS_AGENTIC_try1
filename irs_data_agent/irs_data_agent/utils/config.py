"""
Configuration utilities for the IRS data agent.

This module provides functions for loading and managing configuration settings
from YAML files or environment variables.
"""

import os
import yaml
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file. If None, will look for config.yaml
                    in the current directory and then in the project root directory.
    
    Returns:
        Dict containing configuration settings.
        
    Raises:
        FileNotFoundError: If the configuration file cannot be found.
    """
    if config_path is None:
        # Try current directory first
        if os.path.exists("config.yaml"):
            config_path = "config.yaml"
        # Then try project root
        elif os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yaml")):
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yaml")
        else:
            logger.warning("No configuration file found. Using empty configuration.")
            return {}
    
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found at {config_path}")
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    try:
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
        
        if config is None:
            logger.warning(f"Empty configuration file at {config_path}")
            config = {}
        
        # Override with environment variables if they exist
        if config:
            for key in list(config.keys()):  # Use list to avoid changing dict during iteration
                env_var = f"IRS_DATA_AGENT_{key.upper()}"
                if env_var in os.environ:
                    env_value = os.environ[env_var]
                    # Try to convert to appropriate type
                    config[key] = _convert_env_value(env_value, config[key])
                    logger.debug(f"Overriding config value for {key} with environment variable {env_var}")
        
        return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {str(e)}")
        raise ValueError(f"Invalid YAML in configuration file: {str(e)}")
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise

def _convert_env_value(value: str, original_value: Any) -> Any:
    """
    Convert environment variable string to appropriate type based on original value.
    
    Args:
        value: String value from environment variable
        original_value: Original value from config file to determine type
        
    Returns:
        Converted value
    """
    if original_value is None:
        return value
    
    original_type = type(original_value)
    
    if original_type == bool:
        return value.lower() in ('true', 'yes', '1', 'y')
    elif original_type == int:
        return int(value)
    elif original_type == float:
        return float(value)
    elif original_type == list:
        return value.split(',')
    else:
        return value
