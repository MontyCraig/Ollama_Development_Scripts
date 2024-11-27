# Output Handler Module

Manages test result storage, organization, and retrieval.

## Components

### OutputHandler Class

```python
class OutputHandler:
    """
    Manages test result storage and organization.
    
    Attributes:
        base_path (Path): Base directory for outputs
        format_handlers (Dict[str, Callable]): Output formatters
    """
```

### Key Features

1. **Storage Management**
   * File organization
   * Version control
   * Format conversion
   * Cleanup routines

2. **Data Formats**
   * JSON output
   * Text reports
   * Metrics data
   * Log files

3. **Result Organization**
   * Hierarchical storage
   * Metadata indexing
   * Search capabilities
   * Backup management

### API Reference

#### Storage Operations
```python
def save_result(
    result: TestResult,
    format: str = "json"
) -> Path:
    """Save test result to storage."""

def load_result(
    result_id: str
) -> TestResult:
    """Load test result from storage."""
```

#### Organization
```python
def create_test_directory(
    test_id: str
) -> Path:
    """Create directory structure for test."""

def cleanup_old_results(
    max_age: int = 30
) -> None:
    """Clean up old test results."""
```

### Directory Structure

```
outputs/
├── tests/
│   ├── YYYY-MM-DD/
│   │   ├── test_id/
│   │   │   ├── results.json
│   │   │   ├── metrics.json
│   │   │   └── logs/
├── reports/
│   └── summaries/
└── backups/
```

### Configuration

```yaml
output_handler:
  base_path: "./outputs"
  backup_interval: 86400
  cleanup_age: 2592000
  compression: true
``` 