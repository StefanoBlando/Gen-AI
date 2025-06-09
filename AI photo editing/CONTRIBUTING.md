# Contributing to AI Photo Editor

Thank you for your interest in contributing to AI Photo Editor! This document provides guidelines and information for contributors.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- CUDA-capable GPU (recommended for testing)

### Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/ai-photo-editor.git
   cd ai-photo-editor
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

5. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Development Workflow

### Creating a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### Making Changes

1. Make your changes in the appropriate modules
2. Add or update tests as needed
3. Update documentation if required
4. Ensure your code follows the project style guidelines

### Testing

Run the full test suite:
```bash
python -m pytest
```

Run specific tests:
```bash
python -m pytest tests/test_segmentation.py
python -m pytest tests/test_inpainting.py -v
```

Run tests with coverage:
```bash
python -m pytest --cov=src tests/
```

### Code Quality

The project uses several tools to maintain code quality:

#### Code Formatting
```bash
black src/ tests/
```

#### Linting
```bash
flake8 src/ tests/
```

#### Type Checking
```bash
mypy src/
```

#### Pre-commit Hooks
All checks run automatically on commit:
```bash
git commit -m "Your commit message"
```

## Code Guidelines

### Python Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Maximum line length: 88 characters (Black default)
- Use type hints for function signatures
- Write docstrings for all public functions and classes

### Documentation

- Add docstrings using Google style:
  ```python
  def example_function(param1: str, param2: int) -> bool:
      """Brief description of the function.
      
      Args:
          param1: Description of param1.
          param2: Description of param2.
          
      Returns:
          Description of return value.
          
      Raises:
          ValueError: Description of when this exception is raised.
      """
  ```

- Update README.md if adding new features
- Add examples for new functionality

### Testing

- Write unit tests for new functions
- Include integration tests for complex workflows
- Use meaningful test names that describe what is being tested
- Mock external dependencies (model downloads, GPU operations)

Example test structure:
```python
def test_sam_processor_segments_correctly():
    """Test that SAM processor generates valid segmentation masks."""
    # Arrange
    processor = SAMProcessor()
    test_image = create_test_image()
    test_points = [[100, 100]]
    
    # Act
    mask = processor.segment(test_image, test_points)
    
    # Assert
    assert mask is not None
    assert mask.shape == test_image.size[::-1]
```

## Project Structure

```
src/
├── segmentation/     # SAM integration and utilities
├── inpainting/       # Diffusion model integration
├── ui/              # Gradio interface and web components
├── utils/           # Shared utilities and configuration
└── workflows/       # Advanced editing workflows

tests/               # Test suite
├── test_segmentation.py
├── test_inpainting.py
└── test_integration.py

examples/            # Usage examples and demos
docs/               # Documentation
```

## Commit Guidelines

### Commit Message Format

Use the conventional commit format:

```
type(scope): brief description

Optional longer description explaining the change in more detail.

Fixes #issue-number
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring without functionality changes
- `test`: Adding or updating tests
- `chore`: Maintenance tasks, dependency updates

### Examples

```
feat(segmentation): add batch processing support

Add ability to process multiple images in a single call to improve
performance for batch operations.

Fixes #123
```

```
fix(inpainting): resolve memory leak in iterative workflows

The iterative editor was not properly cleaning up GPU memory between
iterations, causing OOM errors on longer sessions.

Fixes #456
```

## Pull Request Process

1. Ensure your branch is up to date with main:
   ```bash
   git checkout main
   git pull upstream main
   git checkout your-feature-branch
   git rebase main
   ```

2. Run the full test suite and ensure all tests pass
3. Update documentation if needed
4. Create a pull request with:
   - Clear title and description
   - Reference to related issues
   - Screenshots for UI changes
   - Performance impact notes if applicable

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

## Issue Reporting

### Bug Reports

Include:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, GPU)
- Error messages and stack traces
- Minimal code example if possible

### Feature Requests

Include:
- Clear description of the proposed feature
- Use case and motivation
- Possible implementation approach
- Examples of similar features in other projects

## Performance Considerations

- Profile code changes that might affect performance
- Consider memory usage, especially for GPU operations
- Test with different image sizes and batch sizes
- Document performance implications in pull requests

## Getting Help

- Check existing issues and discussions
- Ask questions in GitHub Discussions
- Join our community chat (if available)
- Contact maintainers for major changes

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes for significant contributions
- Special recognition for major features or fixes

Thank you for contributing to AI Photo Editor!
