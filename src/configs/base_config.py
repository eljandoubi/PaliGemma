"""Base config class"""

from typing import Dict, Any
from dataclasses import fields

class BaseConfig:
    """Base Config Class"""
    @classmethod
    def from_dict(cls, data: Dict[str,Any]):
        """Load fields from config"""
        # Get the set of valid fields for the dataclass
        valid_fields = {f.name for f in fields(cls)}
        
        # Filter the input dictionary to only include valid fields
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        
        # Instantiate the dataclass using the filtered data
        return cls(**filtered_data)