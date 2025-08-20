"""Configuration management for the Mean Reversion Trading System."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv


class Config:
    """Configuration manager that loads settings from YAML files and environment variables."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. Defaults to configs/config.yaml
        """
        # Load environment variables
        load_dotenv()
        
        # Set default config path
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file with environment variable substitution."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as file:
            config_content = file.read()
        
        # Substitute environment variables
        config_content = self._substitute_env_vars(config_content)
        
        # Parse YAML
        config = yaml.safe_load(config_content)
        
        return config
    
    def _substitute_env_vars(self, content: str) -> str:
        """Substitute environment variables in configuration content."""
        import re
        
        def replace_env_var(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) else ""
            return os.getenv(var_name, default_value)
        
        # Pattern to match ${VAR_NAME} or ${VAR_NAME:default_value}
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
        return re.sub(pattern, replace_env_var, content)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'data.providers.yahoo_finance.enabled')
            default: Default value if key is not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def reload(self) -> None:
        """Reload configuration from file."""
        self._config = self._load_config()
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()
    
    # Convenience properties for commonly used configurations
    @property
    def data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.get('data', {})
    
    @property
    def strategy_config(self) -> Dict[str, Any]:
        """Get strategy configuration."""
        return self.get('strategies', {})
    
    @property
    def risk_config(self) -> Dict[str, Any]:
        """Get risk management configuration."""
        return self.get('risk_management', {})
    
    @property
    def backtesting_config(self) -> Dict[str, Any]:
        """Get backtesting configuration."""
        return self.get('backtesting', {})
    
    @property
    def logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.get('logging', {})


# Global configuration instance
_config_instance = None


def get_config() -> Config:
    """Get global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance
