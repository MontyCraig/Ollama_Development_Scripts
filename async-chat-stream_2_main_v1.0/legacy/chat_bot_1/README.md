# Chat Report Generator

## Overview

The Chat Report Generator is a Python script that converts JSON chat files into readable text reports. It processes chat history files, preserving model information and conversation content while providing a clean, formatted output.

## Features

* Automatically processes new and modified JSON chat files
* Skips already processed files that haven't changed
* Handles multiple JSON formats (both dictionary and list structures)
* Provides detailed logging of operations
* Creates organized, readable text reports
* Maintains file timestamps for efficient processing

## Requirements

* Python 3.6+
* Standard library modules:
  * json
  * datetime
  * pathlib
  * logging
  * typing

## Installation


1. Place `create_txt_report.py` in your chat history directory
2. Ensure execute permissions:

```

## Class Structure

### ChatReportGenerator
Main class handling the conversion process.

#### Methods:
- `__init__(input_dir: str, output_dir: str)`: Initializes the generator
- `get_existing_reports() -> Set[str]`: Gets list of processed files
- `needs_processing(json_file: Path, existing_reports: Set[str]) -> bool`: Checks if file needs processing
- `load_json_file(file_path: Path) -> Dict`: Loads and parses JSON files
- `format_chat_message(message: Dict) -> str`: Formats individual messages
- `generate_report_content(data: Union[Dict, List], filename: str) -> str`: Generates report content
- `process_files() -> None`: Main processing method

## Report Format
Reports are generated with the following structure:

## Usage
Run the script from the command line:

```

The script will:
1. Scan the current directory for JSON chat files
2. Create a 'reports' directory in the parent folder
3. Generate text reports for new or modified chat files
4. Skip processing of unchanged files

## File Structure



## Logging
The script provides detailed logging:
- INFO level: Normal operation information
- DEBUG level: Detailed processing information
- ERROR level: Error messages with stack traces

## Error Handling
- JSON parsing errors
- Unicode decode errors
- File access errors
- General exception handling with detailed logging

## Development
- Version: 1.1
- Author: Assistant
- License: MIT

## Future Improvements
Potential areas for enhancement:
- Add command line arguments for input/output directories
- Support for additional chat file formats
- Customizable report templates
- Batch processing options
- Statistical analysis of chat data
- Export to additional formats (CSV, HTML, etc.)

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## Support
For issues, questions, or contributions, please:
1. Check existing issues
2. Create a new issue with a detailed description
3. Provide relevant log output and file samples

## License
This project is licensed under the MIT License - see the LICENSE file for details.
