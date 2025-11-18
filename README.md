# Project Title

PySCIn - Python Systematic Coverage Information

[![Python Version](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/release/python-3130/)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Issues](https://img.shields.io/github/issues/trenod/PySCIn)](https://github.com/trenod/PySCIn/issues)

A tool for generating execution traces and performing coverage analysis on Python codebases that use doctest.

## 📖 About The Project

This project is designed to facilitate the analysis of Python codebases by generating detailed execution traces and performing coverage analysis. It is particularly useful for projects that utilize doctest for testing, allowing developers to gain insights into code execution paths, performance bottlenecks, and overall code coverage.

**Key Features:**
* Generates detailed execution traces of Python scripts.
* Analyzes code for performance bottlenecks.
* Customizable analysis with options for storage limits and timeouts.
* Designed to work with Python 3.13 and later versions.
* Specifically tailored for projects that utilize doctest for testing.
* Example usage with the Python Algorithms repository.
* Modular design for easy extension and integration.
* Comprehensive logging and error handling.
* Easy to set up and run with minimal dependencies.
* Provides insights into code coverage and execution paths.
* Suitable for both development and research purposes.
* Includes example scripts and datasets for quick testing.

## 🛠️ Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Make sure you have the following software installed on your system:
* **Git:** To clone the repositories.
* **Python 3.13** or later.


### Installation

1.  **Clone the main repository:**
    ```bash
    git clone [https://github.com/trenod/PySCIn.git](https://github.com/trenod/PySCIn.git)
    ```
2.  **Navigate into the project directory:**
    ```bash
    cd PySCIn
    ```
3.  **Clone the target Python Algorithms repository:**
    This repository will be used as the source code for the analysis. It should be cloned into the same folder as this project.
    ```bash
    git clone [https://github.com/TheAlgorithms/Python](https://github.com/TheAlgorithms/Python)
    ```
4.  **Navigate into the Python Algorithms directory:**
    ```bash
    cd Python
    ```
5.  **Install any dependencies for the Python Algorithms repository if necessary.**
   ```bash
   pip install -r requirements.txt
   ```
---

## 🚀 Usage

To run the pipeline, execute the `Main.py` script from the root directory of the project. You must provide the path to the target folder you wish to analyze.

To run the unit tests, use the following command from the root directory:

```bash
pytest tests/TestMCDCAnalyzer.py
```

### Example

Here is an example command that runs the pipeline on the `maths` examples from the Python Algorithms repository, witha limitation for the trace files of max 2GB and a timeout for individual analyses of 10 minutes, and using the built-in eval function.

```bash
python3.13 Main.py ./Python/maths/ --limit-mb 2000 --timeout-min 10 --use-builtin-eval
```