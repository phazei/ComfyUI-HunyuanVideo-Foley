"""Tests for configuration utilities."""

import pytest
import tempfile
import yaml
from pathlib import Path

from hunyuanvideo_foley.utils.config_utils import AttributeDict, load_yaml


class TestAttributeDict:
    """Test cases for AttributeDict class."""

    def test_dict_access(self):
        """Test dictionary-style access."""
        data = {"key1": "value1", "key2": {"nested": "value2"}}
        attr_dict = AttributeDict(data)
        
        assert attr_dict["key1"] == "value1"
        assert attr_dict["key2"]["nested"] == "value2"
    
    def test_attribute_access(self):
        """Test attribute-style access.""" 
        data = {"key1": "value1", "key2": {"nested": "value2"}}
        attr_dict = AttributeDict(data)
        
        assert attr_dict.key1 == "value1"
        assert attr_dict.key2.nested == "value2"
    
    def test_list_handling(self):
        """Test list data handling."""
        data = [1, 2, {"nested": "value"}]
        attr_dict = AttributeDict(data)
        
        assert attr_dict[0] == 1
        assert attr_dict[2].nested == "value"
    
    def test_keys_method(self):
        """Test keys() method."""
        data = {"key1": "value1", "key2": "value2"}
        attr_dict = AttributeDict(data)
        
        keys = list(attr_dict.keys())
        assert "key1" in keys
        assert "key2" in keys
    
    def test_get_method(self):
        """Test get() method."""
        data = {"key1": "value1"}
        attr_dict = AttributeDict(data)
        
        assert attr_dict.get("key1") == "value1"
        assert attr_dict.get("nonexistent", "default") == "default"


class TestLoadYaml:
    """Test cases for load_yaml function."""
    
    def test_load_valid_yaml(self):
        """Test loading valid YAML file."""
        data = {"model": {"name": "test_model", "params": {"lr": 0.001}}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(data, f)
            yaml_path = f.name
        
        try:
            result = load_yaml(yaml_path)
            assert result.model.name == "test_model"
            assert result.model.params.lr == 0.001
        finally:
            Path(yaml_path).unlink()
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_yaml("nonexistent.yaml")
    
    def test_load_invalid_yaml(self):
        """Test loading invalid YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [\n")  # Invalid YAML
            yaml_path = f.name
        
        try:
            with pytest.raises(yaml.YAMLError):
                load_yaml(yaml_path)
        finally:
            Path(yaml_path).unlink()