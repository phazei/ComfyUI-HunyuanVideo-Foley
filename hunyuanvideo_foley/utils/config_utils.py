"""Configuration utilities for the HunyuanVideo-Foley project."""

import yaml
from pathlib import Path
from typing import Any, Dict, List, Union

class AttributeDict:
    
    def __init__(self, data: Union[Dict, List, Any]):
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    value = AttributeDict(value)
                setattr(self, self._sanitize_key(key), value)
        elif isinstance(data, list):
            self._list = [AttributeDict(item) if isinstance(item, (dict, list)) else item 
                         for item in data]
        else:
            self._value = data
    
    def _sanitize_key(self, key: str) -> str:
        import re
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', str(key))
        if sanitized[0].isdigit():
            sanitized = f'_{sanitized}'
        return sanitized
    
    def __getitem__(self, key):
        if hasattr(self, '_list'):
            return self._list[key]
        return getattr(self, self._sanitize_key(key))
    
    def __setitem__(self, key, value):
        if hasattr(self, '_list'):
            self._list[key] = value
        else:
            setattr(self, self._sanitize_key(key), value)
    
    def __iter__(self):
        if hasattr(self, '_list'):
            return iter(self._list)
        return iter(self.__dict__.keys())
    
    def __len__(self):
        if hasattr(self, '_list'):
            return len(self._list)
        return len(self.__dict__)
    
    def get(self, key, default=None):
        try:
            return self[key]
        except (KeyError, AttributeError, IndexError):
            return default
    
    def keys(self):
        if hasattr(self, '_list'):
            return range(len(self._list))
        elif hasattr(self, '_value'):
            return []
        else:
            return [key for key in self.__dict__.keys() if not key.startswith('_')]
    
    def values(self):
        if hasattr(self, '_list'):
            return self._list
        elif hasattr(self, '_value'):
            return [self._value]
        else:
            return [value for key, value in self.__dict__.items() if not key.startswith('_')]
    
    def items(self):
        if hasattr(self, '_list'):
            return enumerate(self._list)
        elif hasattr(self, '_value'):
            return []
        else:
            return [(key, value) for key, value in self.__dict__.items() if not key.startswith('_')]
    
    def __repr__(self):
        if hasattr(self, '_list'):
            return f"AttributeDict({self._list})"
        elif hasattr(self, '_value'):
            return f"AttributeDict({self._value})"
        return f"AttributeDict({dict(self.__dict__)})"
    
    def to_dict(self) -> Union[Dict, List, Any]:
        if hasattr(self, '_list'):
            return [item.to_dict() if isinstance(item, AttributeDict) else item 
                   for item in self._list]
        elif hasattr(self, '_value'):
            return self._value
        else:
            result = {}
            for key, value in self.__dict__.items():
                if isinstance(value, AttributeDict):
                    result[key] = value.to_dict()
                else:
                    result[key] = value
            return result

def load_yaml(file_path: str, encoding: str = 'utf-8') -> AttributeDict:
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            data = yaml.safe_load(file)
            return AttributeDict(data)
    except FileNotFoundError:
        raise FileNotFoundError(f"YAML file not found: {file_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"YAML format error: {e}")
