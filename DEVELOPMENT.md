# Development Guide

This document provides guidelines for developing and contributing to the HunyuanVideo-Foley project.

## Code Style and Quality

### Code Formatting

We use the following tools to maintain consistent code style:

- **Black**: Code formatter with 120 character line length
- **isort**: Import sorter compatible with Black
- **flake8**: Linting and style checking
- **mypy**: Static type checking

### Pre-commit Hooks

Install pre-commit hooks to automatically format code before commits:

```bash
pip install pre-commit
pre-commit install
```

### Manual Code Formatting

Format code manually:

```bash
# Format all Python files
black --line-length 120 .

# Sort imports
isort --profile black --line-length 120 .

# Check code style
flake8 --max-line-length 120

# Type checking
mypy --ignore-missing-imports .
```

## Project Structure

```
hunyuanvideo_foley/
├── models/                 # Model implementations
│   ├── hifi_foley.py      # Main model
│   ├── nn/                # Neural network layers
│   ├── dac_vae/           # Audio VAE
│   └── synchformer/       # Synchronization model
├── utils/                 # Utilities
│   ├── config_utils.py    # Configuration handling
│   ├── feature_utils.py   # Feature extraction
│   ├── model_utils.py     # Model loading/saving
│   └── media_utils.py     # Audio/video processing
└── constants.py           # Project constants
```

## Coding Standards

### Error Handling

- Use custom exceptions for domain-specific errors
- Always validate inputs at function boundaries
- Log errors with appropriate levels (ERROR, WARNING, INFO)
- Provide helpful error messages to users

### Type Hints

- Add type hints to all function parameters and return values
- Use `Optional[Type]` for nullable parameters
- Import types from `typing` module

### Documentation

- Add docstrings to all public functions and classes
- Use Google-style docstrings
- Document parameters, return values, and exceptions

### Example Function

```python
def process_video(
    video_path: str,
    max_duration: Optional[float] = None
) -> Tuple[np.ndarray, float]:
    """
    Process video file and extract frames.
    
    Args:
        video_path: Path to input video file
        max_duration: Maximum duration in seconds (optional)
        
    Returns:
        Tuple of (frames array, duration in seconds)
        
    Raises:
        FileNotFoundError: If video file doesn't exist
        VideoProcessingError: If video processing fails
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Implementation here...
```

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_feature_utils.py

# Run with coverage
python -m pytest --cov=hunyuanvideo_foley
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files as `test_*.py`
- Use descriptive test function names
- Test edge cases and error conditions

## Development Workflow

1. **Setup Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   
   pip install -r requirements.txt
   pip install -e .
   ```

2. **Install Development Tools**
   ```bash
   pre-commit install
   ```

3. **Make Changes**
   - Follow the coding standards above
   - Add tests for new functionality
   - Update documentation as needed

4. **Run Quality Checks**
   ```bash
   black --check --line-length 120 .
   isort --check-only --profile black .
   flake8 --max-line-length 120
   mypy --ignore-missing-imports .
   pytest
   ```

5. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

## Performance Considerations

- Use `torch.no_grad()` for inference-only code
- Leverage GPU when available
- Implement batch processing where possible
- Profile code to identify bottlenecks

## Dependencies

- Keep dependencies minimal and well-maintained
- Pin versions for reproducibility
- Separate development dependencies from runtime dependencies
- Document any special installation requirements

## Configuration

- Use centralized configuration in `constants.py`
- Support environment variable overrides
- Provide sensible defaults for all parameters
- Validate configuration at startup