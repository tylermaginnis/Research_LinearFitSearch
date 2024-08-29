# Search Algorithm Performance Comparison

This project compares the performance of three different search algorithms: Linear Fit Search, Binary Search, and Hybrid Search. The comparison is based on various distributions of data, including random values, prime numbers, Fibonacci sequence, geometric progression, random gaps, and exponential growth.

## Table of Contents

- [Introduction](#introduction)
- [Algorithms](#algorithms)
  - [Linear Fit Search](#linear-fit-search)
  - [Binary Search](#binary-search)
  - [Hybrid Search](#hybrid-search)
- [Test Data](#test-data)
- [Metrics](#metrics)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The goal of this project is to evaluate the performance of different search algorithms on various types of data distributions. The performance is measured in terms of execution time and memory usage. The results are normalized to provide a fair comparison between the algorithms.

## Algorithms

### Linear Fit Search

Linear Fit Search is a simple search algorithm that iterates through the list of values to find the target value. It is efficient for small datasets but can be slow for larger datasets.

### Binary Search

Binary Search is a more efficient search algorithm that works on sorted lists. It repeatedly divides the search interval in half until the target value is found or the interval is empty.

### Hybrid Search

Hybrid Search is a combination of Linear Fit Search and Binary Search. It uses Linear Fit Search for small datasets and switches to Binary Search for larger datasets.

## Test Data

The test data consists of various distributions to evaluate the performance of the search algorithms:

1. Random values
2. Prime numbers
3. Fibonacci sequence
4. Geometric progression
5. Random gaps
6. Exponential growth

Each distribution is tested with multiple target values to ensure a comprehensive evaluation.

## Metrics

The performance of the search algorithms is measured using the following metrics:

- **Execution Time**: The time taken to find the target value.
- **Memory Usage**: The amount of memory used during the search process.

The metrics are normalized to provide a fair comparison between the algorithms.

## Results

The results of the performance comparison are printed to the console. The algorithm with the shortest normalized execution time is declared the winner for each test case. The total wins for each algorithm are also displayed.

## Usage

To run the performance comparison, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/search-algorithm-comparison.git
   ```
2. Navigate to the project directory:
   ```sh
   cd search-algorithm-comparison
   ```
3. Build and run the project:
   ```sh
   cargo run
   ```

The results will be displayed in the console.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, please create a pull request or open an issue.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
