# Search Algorithm Performance Comparison

This project aims to compare the performance of several different search algorithms: Linear Fit Search, Binary Search, Interpolated Binary Search, Hybrid Search, Hybrid Interpolated Search, and Hybrid Adaptive Search. The comparison is based on various distributions of data, including random values, prime numbers, Fibonacci sequence, geometric progression, random gaps, and exponential growth.

## Credits

This project was inspired by the [Linear Fit Search](https://blog.demofox.org/2019/03/22/linear-fit-search/) article on the Demofox blog.

## Table of Contents

- [Introduction](#introduction)
- [Algorithms](#algorithms)
  - [Linear Fit Search](#linear-fit-search)
  - [Binary Search](#binary-search)
  - [Interpolated Binary Search](#interpolated-binary-search)
  - [Hybrid Search](#hybrid-search)
  - [Hybrid Interpolated Search](#hybrid-interpolated-search)
  - [Hybrid Adaptive Search](#hybrid-adaptive-search)
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

Linear Fit Search is an algorithm that attempts to find the target value by leveraging a linear fit equation. It starts by calculating the initial slope (m) and intercept (b) based on the minimum and maximum values in the list. The algorithm then iteratively refines the search range by updating the slope and intercept, and calculating a guess index using the linear fit equation. This process continues until the target value is found or the search range is exhausted. Linear Fit Search is efficient for datasets where the values are approximately linear, but its performance may vary for other types of distributions. For more details, refer to the implementation in [main.rs](src/main.rs).

### Binary Search

Binary Search is a more efficient search algorithm that works on sorted lists. It repeatedly divides the search interval in half until the target value is found or the interval is empty. The algorithm starts by comparing the target value to the middle element of the list. If the target value is equal to the middle element, the search is complete. If the target value is less than the middle element, the algorithm continues the search in the lower half of the list. If the target value is greater than the middle element, the algorithm continues the search in the upper half of the list. This process is repeated until the target value is found or the search interval is empty. Binary Search has a logarithmic time complexity of O(log n), making it much faster than Linear Fit Search for larger datasets.

### Interpolated Binary Search

Interpolated Binary Search is an enhancement of Binary Search that works well on uniformly distributed data. Instead of always choosing the middle element, it estimates the position of the target value based on the values at the low and high indices. This can reduce the number of iterations needed to find the target value. The algorithm calculates the estimated position using the formula:
\[ \text{pos} = \text{low} + \left( \frac{(\text{target} - \text{values}[\text{low}]) \times (\text{high} - \text{low})}{\text{values}[\text{high}] - \text{values}[\text{low}]} \right) \]
If the estimated position is correct, the search is complete. Otherwise, the algorithm continues the search in the appropriate half of the list. For more details, refer to the implementation in [main.rs](src/main.rs).

### Hybrid Search

Hybrid Search is a combination of Linear Fit Search and Binary Search. It uses Linear Fit Search for small datasets and switches to Binary Search for larger datasets. The idea is to take advantage of the simplicity of Linear Fit Search for small datasets while leveraging the efficiency of Binary Search for larger datasets. The threshold for switching between the two algorithms can be determined based on the size of the dataset and the specific use case.

### Hybrid Interpolated Search

Hybrid Interpolated Search combines the principles of Interpolated Binary Search and Linear Fit Search. It starts with Linear Fit Search and periodically adjusts the threshold based on the variance of the data. If the variance is low, it uses Interpolated Binary Search to estimate the position of the target value. This approach aims to balance the benefits of both algorithms, providing efficient search performance for a wide range of data distributions. For more details, refer to the implementation in [main.rs](src/main.rs).

### Hybrid Adaptive Search

Hybrid Adaptive Search is an advanced search algorithm that adapts its strategy based on the characteristics of the data. It starts with a combination of Linear Fit Search and Binary Search, similar to Hybrid Search. However, it periodically adjusts the threshold and search strategy based on the variance and distribution of the data. This adaptive approach allows the algorithm to dynamically switch between different search methods, optimizing performance for various data distributions. For more details, refer to the implementation in [main.rs](src/main.rs).

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


## Analysis of results.txt

The results from `results.txt` provide a detailed comparison of various search algorithms across different test cases. Each test case evaluates the performance of the algorithms on different data distributions, including random values, prime numbers, Fibonacci sequence, geometric progression, random gaps, and exponential growth.

Key observations from the results:

1. **Hybrid Linear-Binary Optimized Threshold Search**:
   - Frequently emerges as the winner in many test cases.
   - Demonstrates superior performance in terms of execution time, especially for non-linear and difficult distributions.
   - Consistently shows low memory usage across different data distributions.

2. **Hybrid Linear-Adaptive Binary Search**:
   - Performs well in cases with mixed distributions and non-linear ranges.
   - Shows competitive execution times and memory usage, often close to the optimized threshold search.

3. **Hybrid Linear-Interpolated Binary Search**:
   - Exhibits strong performance in test cases with non-linear and mixed distributions.
   - Execution times are generally competitive, but not as consistently low as the optimized threshold search.

4. **Linear Fit Search**:
   - Performs well in specific cases, particularly with small ranges and repeated blocks.
   - Execution times and memory usage are higher compared to hybrid search algorithms in most cases.

5. **Binary Search**:
   - Shows reliable performance with predictable execution times and memory usage.
   - Generally outperformed by hybrid search algorithms in terms of execution time.

6. **Interpolated Binary Search**:
   - Demonstrates good performance in cases with sorted data and specific mathematical properties.
   - Execution times are competitive, but memory usage is similar to other binary search variants.

Overall, the results highlight the effectiveness of hybrid search algorithms, particularly the Hybrid Linear-Binary Optimized Threshold Search, in handling various data distributions efficiently. The detailed metrics for each test case provide valuable insights into the strengths and weaknesses of each algorithm, guiding the selection of the most suitable search method for different scenarios.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, please create a pull request or open an issue. Contributions can include bug fixes, new features, performance improvements, or updates to the documentation.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. The MIT License allows for free use, modification, and distribution of the software, provided that the original copyright notice and permission notice are included in all copies or substantial portions of the software. The software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.
