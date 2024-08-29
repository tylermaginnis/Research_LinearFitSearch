# Search Algorithm Performance Comparison

This project aims to compare the performance of three different search algorithms: Linear Fit Search, Binary Search, and Hybrid Search. The comparison is based on various distributions of data, including random values, prime numbers, Fibonacci sequence, geometric progression, random gaps, and exponential growth.

## Credits

This project was inspired by the [Linear Fit Search](https://blog.demofox.org/2019/03/22/linear-fit-search/) article on the Demofox blog.

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

Linear Fit Search is a simple search algorithm that iterates through the list of values to find the target value. It starts from the beginning of the list and checks each element one by one until it finds the target value or reaches the end of the list. This algorithm is efficient for small datasets because it has a linear time complexity of O(n), where n is the number of elements in the list. However, it can be slow for larger datasets because it may need to check many elements before finding the target value.

### Binary Search

Binary Search is a more efficient search algorithm that works on sorted lists. It repeatedly divides the search interval in half until the target value is found or the interval is empty. The algorithm starts by comparing the target value to the middle element of the list. If the target value is equal to the middle element, the search is complete. If the target value is less than the middle element, the algorithm continues the search in the lower half of the list. If the target value is greater than the middle element, the algorithm continues the search in the upper half of the list. This process is repeated until the target value is found or the search interval is empty. Binary Search has a logarithmic time complexity of O(log n), making it much faster than Linear Fit Search for larger datasets.

### Hybrid Search

Hybrid Search is a combination of Linear Fit Search and Binary Search. It uses Linear Fit Search for small datasets and switches to Binary Search for larger datasets. The idea is to take advantage of the simplicity of Linear Fit Search for small datasets while leveraging the efficiency of Binary Search for larger datasets. The threshold for switching between the two algorithms can be determined based on the size of the dataset and the specific use case.

## Test Data

The test data consists of various distributions to evaluate the performance of the search algorithms. Each distribution is designed to test the algorithms under different conditions and data patterns:

1. **Random values**: A list of values generated randomly. This distribution tests the algorithms' performance on unsorted and unpredictable data.
2. **Prime numbers**: A list of prime numbers. This distribution tests the algorithms' performance on a sorted list of unique values with specific mathematical properties.
3. **Fibonacci sequence**: A list of values from the Fibonacci sequence. This distribution tests the algorithms' performance on a sorted list of values with a specific growth pattern.
4. **Geometric progression**: A list of values generated by a geometric progression. This distribution tests the algorithms' performance on a sorted list of values with an exponential growth pattern.
5. **Random gaps**: A list of values with random gaps between consecutive elements. This distribution tests the algorithms' performance on a sorted list of values with irregular intervals.
6. **Exponential growth**: A list of values generated by an exponential function. This distribution tests the algorithms' performance on a sorted list of values with a rapid growth pattern.

Each distribution is tested with multiple target values to ensure a comprehensive evaluation of the algorithms' performance.

## Metrics

The performance of the search algorithms is measured using the following metrics:

- **Execution Time**: The time taken to find the target value. This metric is measured in nanoseconds and provides an indication of the algorithm's speed.
- **Memory Usage**: The amount of memory used during the search process. This metric is measured in bytes and provides an indication of the algorithm's memory efficiency.

The metrics are normalized to provide a fair comparison between the algorithms. Normalization involves dividing the execution time and memory usage by the number of elements in the dataset, resulting in a per-element metric that allows for a more accurate comparison across different dataset sizes.

## Results

The results of the performance comparison are printed to the console. For each test case, the algorithm with the shortest normalized execution time is declared the winner. The total wins for each algorithm are also displayed, providing an overall summary of the performance comparison. The results help identify which algorithm performs best under different conditions and data distributions.

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

The results will be displayed in the console, showing the performance metrics and the winner for each test case.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, please create a pull request or open an issue. Contributions can include bug fixes, new features, performance improvements, or updates to the documentation.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. The MIT License allows for free use, modification, and distribution of the software, provided that the original copyright notice and permission notice are included in all copies or substantial portions of the software. The software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.
