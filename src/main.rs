use std::time::Instant;
use once_cell::sync::Lazy;
use std::sync::Mutex;
use std::collections::HashMap;
use colored::*; // Add colored crate for colorization

static CPU_METRICS: Lazy<Mutex<HashMap<&'static str, Vec<u128>>>> = Lazy::new(|| Mutex::new(HashMap::new()));
static MEMORY_METRICS: Lazy<Mutex<HashMap<&'static str, Vec<usize>>>> = Lazy::new(|| Mutex::new(HashMap::new()));
// Linear Fit Search implementation
fn linear_fit_search(values: &[i32], target: i32) -> (bool, usize, usize) {
    // If the input array is empty, return immediately
    if values.is_empty() {
        return (false, 0, 0);
    }

    // Initialize the minimum and maximum indices and their corresponding values
    let mut min_index = 0;
    let mut max_index = values.len() - 1;
    let mut min = values[min_index];
    let mut max = values[max_index];

    // Initialize the number of guesses
    let mut guesses = 0;

    // Check if the target is outside the range of the array
    if target < min {
        println!("Linear Fit Search iterations: {}", guesses);
        return (false, min_index, guesses);
    }
    if target > max {
        println!("Linear Fit Search iterations: {}", guesses);
        return (false, max_index, guesses);
    }
    // Check if the target is at the boundaries of the array
    if target == min {
        println!("Linear Fit Search iterations: {}", guesses);
        return (true, min_index, guesses);
    }
    if target == max {
        println!("Linear Fit Search iterations: {}", guesses);
        return (true, max_index, guesses);
    }

    // Calculate the initial slope (m) and intercept (b) for the linear fit
    let mut m = (max - min) as f32 / (max_index - min_index) as f32;
    let mut b = min as f32 - m * min_index as f32;

    // Loop until the target is found or the search space is exhausted
    loop {
        guesses += 1;
        // Calculate the guess index using the linear fit equation
        let guess_index = ((target as f32 - b) / m).round() as usize;
        // Clamp the guess index to be within the valid range
        let guess_index = guess_index.clamp(min_index + 1, max_index - 1);
        let guess = values[guess_index];

        // Check if the guess is the target
        if guess == target {
            println!("Linear Fit Search iterations: {}", guesses);
            return (true, guess_index, guesses);
        }

        // Update the search range based on the guess
        if guess < target {
            min_index = guess_index;
            min = guess;
        } else {
            max_index = guess_index;
            max = guess;
        }

        // If the search range is exhausted, return the result
        if min_index + 1 >= max_index {
            println!("Linear Fit Search iterations: {}", guesses);
            return (false, min_index, guesses);
        }

        // Recalculate the slope (m) and intercept (b) for the new search range
        m = (max - min) as f32 / (max_index - min_index) as f32;
        b = min as f32 - m * min_index as f32;
    }
}

// Binary Search implementation
fn binary_search(values: &[i32], target: i32) -> Option<usize> {
    // Initialize the low and high indices for the search range
    let mut low = 0;
    let mut high = values.len() as i32 - 1;
    let mut iterations = 0;

    // Loop until the search range is exhausted
    while low <= high {
        iterations += 1;
        // Calculate the middle index of the current search range
        let mid = (low + high) / 2;
        let mid_value = values[mid as usize];

        // Check if the middle value is the target
        if mid_value == target {
            println!("Binary Search iterations: {}", iterations);
            return Some(mid as usize);
        } else if mid_value < target {
            // If the middle value is less than the target, search the right half
            low = mid + 1;
        } else {
            // If the middle value is greater than the target, search the left half
            high = mid - 1;
        }
    }
    println!("Binary Search iterations: {}", iterations);
    None
}

fn interpolated_binary_search(values: &[i32], target: i32) -> Option<usize> {
    let mut low = 0;
    let mut high = values.len() as i32 - 1;
    let mut iterations = 0;

    while low <= high && target >= values[low as usize] && target <= values[high as usize] {
        iterations += 1;

        if low == high {
            if values[low as usize] == target {
                println!("Interpolated Binary Search iterations: {}", iterations);
                return Some(low as usize);
            }
            break;
        }

        let pos = low + (((high - low) as f32 / (values[high as usize] - values[low as usize]) as f32) * (target - values[low as usize]) as f32) as i32;

        if values[pos as usize] == target {
            println!("Interpolated Binary Search iterations: {}", iterations);
            return Some(pos as usize);
        }

        if values[pos as usize] < target {
            low = pos + 1;
        } else {
            high = pos - 1;
        }
    }

    println!("Interpolated Binary Search iterations: {}", iterations);
    None
}


// Hybrid linear+binary search with adaptive heuristics
fn hybrid_search(values: &[i32], target: i32) -> Option<usize> {
    let mut low = 0;
    let mut high = values.len() as i32 - 1;
    let mut iterations = 0;

    // Initial threshold calculation
    let mut threshold = (high - low) / 10;

    while low <= high {
        iterations += 1;

        // Adjust threshold periodically
        if iterations % 8 == 0 {
            let range = &values[low as usize..=high as usize];
            let (sum, sum_of_squares, early_exit) = range.iter().try_fold((0i64, 0i64, false), |(sum, sq_sum, _), &x| {
                let x_i64 = x as i64;
                let new_sum = sum + x_i64;
                let new_sq_sum = sq_sum + x_i64 * x_i64;
                let mean = new_sum as f64 / range.len() as f64;
                let variance = (new_sq_sum as f64 / range.len() as f64) - (mean * mean);
                if variance < 1.0 {
                    Err((new_sum, new_sq_sum, true))
                } else {
                    Ok((new_sum, new_sq_sum, false))
                }
            }).unwrap_or_else(|res| res);
            if !early_exit {
                let mean = sum as f64 / range.len() as f64;
                let variance = (sum_of_squares as f64 / range.len() as f64) - (mean * mean);
                threshold = (variance.sqrt() as i32).max(1);
            }
        }

        // If within threshold, use linear search
        if high - low <= threshold {
            return values[low as usize..=high as usize]
                .iter()
                .position(|&x| x == target)
                .map(|pos| pos + low as usize);
        }

        let mid = (low + high) / 2;
        let mid_value = unsafe { *values.get_unchecked(mid as usize) };

        if mid_value == target {
            println!("Hybrid Linear-Binary Optimized Threshold Search iterations: {}", iterations);
            return Some(mid as usize);
        } else if mid_value < target {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    println!("Hybrid Linear-Binary Optimized Threshold Search iterations: {}", iterations);
    None
}

fn hybrid_interpolated_search(values: &[i32], target: i32) -> Option<usize> {
    let mut low = 0i64;
    let mut high = (values.len() - 1) as i64;
    let mut iterations = 0;
    let mut threshold = 10;

    while low <= high {
        iterations += 1;

        // Adjust threshold periodically
        if iterations % 8 == 0 {
            let range = &values[low as usize..=high as usize];
            let (sum, sum_of_squares, early_exit) = range.iter().try_fold((0i64, 0i64, false), |(sum, sq_sum, _), &x| {
                let x_i64 = x as i64;
                let new_sum = sum + x_i64;
                let new_sq_sum = sq_sum + x_i64 * x_i64;
                let mean = new_sum as f64 / range.len() as f64;
                let variance = (new_sq_sum as f64 / range.len() as f64) - (mean * mean);
                if variance < 1.0 {
                    Err((new_sum, new_sq_sum, true))
                } else {
                    Ok((new_sum, new_sq_sum, false))
                }
            }).unwrap_or_else(|res| res);
            if !early_exit {
                let mean = sum as f64 / range.len() as f64;
                let variance = (sum_of_squares as f64 / range.len() as f64) - (mean * mean);
                threshold = (variance.sqrt() as i32).max(1);
            }
        }

        // If within threshold, use linear search
        if high - low <= threshold as i64 {
            return values[low as usize..=high as usize]
                .iter()
                .position(|&x| x == target)
                .map(|pos| pos + low as usize);
        }

        // Interpolated Binary Search
        let mid = if values[low as usize] != values[high as usize] {
            low + ((target as i64 - values[low as usize] as i64) * (high - low)
                / (values[high as usize] as i64 - values[low as usize] as i64))
        } else {
            (low + high) / 2
        };

        // Clamp mid to valid index range
        let mid = mid.clamp(low, high);
        
        let mid_value = unsafe { *values.get_unchecked(mid as usize) };

        if mid_value == target {
            println!("Hybrid Linear-Interpolated Binary Search iterations: {}", iterations);
            return Some(mid as usize);
        } else if mid_value < target {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    println!("Hybrid Linear-Interpolated Binary Search iterations: {}", iterations);
    None
}

fn hybrid_adaptive_search(values: &[i32], target: i32) -> Option<usize> {
    let mut low = 0i64;
    let mut high = (values.len() - 1) as i64;
    let mut iterations = 0;
    let mut threshold = 10;

    while low <= high {
        iterations += 1;

        // Adjust threshold periodically
        if iterations % 8 == 0 {
            let range = &values[low as usize..=high as usize];
            let (sum, sum_of_squares, early_exit) = range.iter().try_fold((0i64, 0i64, false), |(sum, sq_sum, _), &x| {
                let x_i64 = x as i64;
                let new_sum = sum + x_i64;
                let new_sq_sum = sq_sum + x_i64 * x_i64;
                let mean = new_sum as f64 / range.len() as f64;
                let variance = (new_sq_sum as f64 / range.len() as f64) - (mean * mean);
                if variance < 1.0 {
                    Err((new_sum, new_sq_sum, true))
                } else {
                    Ok((new_sum, new_sq_sum, false))
                }
            }).unwrap_or_else(|res| res);
            if !early_exit {
                let mean = sum as f64 / range.len() as f64;
                let variance = (sum_of_squares as f64 / range.len() as f64) - (mean * mean);
                threshold = (variance.sqrt() as i32).max(1);
            }
        }

        // If within threshold, use linear search
        if high - low <= threshold as i64 {
            return values[low as usize..=high as usize]
                .iter()
                .position(|&x| x == target)
                .map(|pos| pos + low as usize);
        }

        // Determine if interpolated search should be used
        let use_interpolation = if values[low as usize] != values[high as usize] {
            let predicted_mid = low + ((target as i64 - values[low as usize] as i64) * (high - low)
                / (values[high as usize] as i64 - values[low as usize] as i64));
            // Use interpolation if predicted mid is closer to target than binary mid
            let binary_mid = (low + high) / 2;
            (predicted_mid - target as i64).abs() < (binary_mid - target as i64).abs()
        } else {
            false
        };

        let mid = if use_interpolation {
            // Interpolated Binary Search
            low + ((target as i64 - values[low as usize] as i64) * (high - low)
                / (values[high as usize] as i64 - values[low as usize] as i64))
        } else {
            // Traditional Binary Search
            (low + high) / 2
        };

        // Clamp mid to valid index range
        let mid = mid.clamp(low, high);
        
        let mid_value = unsafe { *values.get_unchecked(mid as usize) };

        if mid_value == target {
            println!("Hybrid Linear-Adaptive Binary Search iterations: {}", iterations);
            return Some(mid as usize);
        } else if mid_value < target {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    println!("Hybrid Linear-Adaptive Binary Search iterations: {}", iterations);
    None
}








use rand::Rng;

// Define a function to generate test data for benchmarks
fn generate_test_data() -> Vec<(Vec<i32>, i32, Option<usize>, String)> {
    let values = (1..=200).step_by(2).collect::<Vec<i32>>();
    let mut rng = rand::thread_rng();
    let mut random_values = (1..=200).map(|_| rng.gen_range(1..=200)).collect::<Vec<i32>>();
    random_values.sort_unstable();

    let mut test_data = Vec::new();

    // Add edge cases
    test_data.push((values.clone(), 1, Some(0), "First element".to_string())); // First element
    test_data.push((values.clone(), 199, Some(99), "Last element".to_string())); // Last element
    test_data.push((values.clone(), 200, None, "Element not in the list".to_string())); // Element not in the list

    // Add random cases
    for _ in 0..3 {
        let target = random_values[rng.gen_range(0..random_values.len())];
        let expected = random_values.binary_search(&target).ok();
        test_data.push((random_values.clone(), target, expected, "Random case".to_string()));
    }

    // Set 1: Small range of values
    let small_values = (1..=20).collect::<Vec<i32>>();
    for _ in 0..3 {
        let target = small_values[rng.gen_range(0..small_values.len())];
        let expected = small_values.binary_search(&target).ok();
        test_data.push((small_values.clone(), target, expected, "Small range of values".to_string()));
    }

    // Set 2: Larger range of values
    let large_values = (1..=1000).step_by(3).collect::<Vec<i32>>();
    for _ in 0..3 {
        let target = large_values[rng.gen_range(0..large_values.len())];
        let expected = large_values.binary_search(&target).ok();
        test_data.push((large_values.clone(), target, expected, "Larger range of values".to_string()));
    }

    // Set 3: Even larger range of values
    let larger_values = (1..=10000).step_by(5).collect::<Vec<i32>>();
    for _ in 0..3 {
        let target = larger_values[rng.gen_range(0..larger_values.len())];
        let expected = larger_values.binary_search(&target).ok();
        test_data.push((larger_values.clone(), target, expected, "Even larger range of values".to_string()));
    }

    // Set 4: Very large range of values
    let very_large_values = (1..=100000).step_by(7).collect::<Vec<i32>>();
    for _ in 0..3 {
        let target = very_large_values[rng.gen_range(0..very_large_values.len())];
        let expected = very_large_values.binary_search(&target).ok();
        test_data.push((very_large_values.clone(), target, expected, "Very large range of values".to_string()));
    }

    // Set 5: Extremely large range of values
    let extremely_large_values = (1..=1000000).step_by(11).collect::<Vec<i32>>();
    for _ in 0..3 {
        let target = extremely_large_values[rng.gen_range(0..extremely_large_values.len())];
        let expected = extremely_large_values.binary_search(&target).ok();
        test_data.push((extremely_large_values.clone(), target, expected, "Extremely large range of values".to_string()));
    }

    // Set 6: Non-linear range of values (squares)
    let non_linear_values_1 = (1..=100).map(|x| x * x).collect::<Vec<i32>>();
    for _ in 0..3 {
        let target = non_linear_values_1[rng.gen_range(0..non_linear_values_1.len())];
        let expected = non_linear_values_1.binary_search(&target).ok();
        test_data.push((non_linear_values_1.clone(), target, expected, "Non-linear range of values (squares)".to_string()));
    }

    // Set 7: Non-linear range of values (cubes)
    let non_linear_values_2 = (1..=50).map(|x| x * x * x).collect::<Vec<i32>>();
    for _ in 0..3 {
        let target = non_linear_values_2[rng.gen_range(0..non_linear_values_2.len())];
        let expected = non_linear_values_2.binary_search(&target).ok();
        test_data.push((non_linear_values_2.clone(), target, expected, "Non-linear range of values (cubes)".to_string()));
    }

    // Set 8: Non-linear range of values (powers of 2)
    let non_linear_values_3 = (0..=20).map(|x| 2_i32.pow(x)).collect::<Vec<i32>>();
    for _ in 0..3 {
        let target = non_linear_values_3[rng.gen_range(0..non_linear_values_3.len())];
        let expected = non_linear_values_3.binary_search(&target).ok();
        test_data.push((non_linear_values_3.clone(), target, expected, "Non-linear range of values (powers of 2)".to_string()));
    }

    // Set 9: Non-linear range of values (factorials)
    let non_linear_values_4 = (1..=10).scan(1, |state, x| {
        *state = *state * x;
        Some(*state)
    }).collect::<Vec<i32>>();
    for _ in 0..3 {
        let target = non_linear_values_4[rng.gen_range(0..non_linear_values_4.len())];
        let expected = non_linear_values_4.binary_search(&target).ok();
        test_data.push((non_linear_values_4.clone(), target, expected, "Non-linear range of values (factorials)".to_string()));
    }

    // Set 10: Non-linear range of values (Fibonacci sequence)
    let non_linear_values_5 = {
        let mut fib = vec![0, 1];
        for i in 2..20 {
            let next = fib[i - 1] + fib[i - 2];
            fib.push(next);
        }
        fib
    };
    for _ in 0..3 {
        let target = non_linear_values_5[rng.gen_range(0..non_linear_values_5.len())];
        let expected = non_linear_values_5.binary_search(&target).ok();
        test_data.push((non_linear_values_5.clone(), target, expected, "Non-linear range of values (Fibonacci sequence)".to_string()));
    }

    // Set 11: Mixed distribution (linear and squares)
    let mixed_values_1 = (1..=50).flat_map(|x| vec![x, x * x]).collect::<Vec<i32>>();
    for _ in 0..3 {
        let target = mixed_values_1[rng.gen_range(0..mixed_values_1.len())];
        let expected = mixed_values_1.binary_search(&target).ok();
        test_data.push((mixed_values_1.clone(), target, expected, "Mixed distribution (linear and squares)".to_string()));
    }

    // Set 12: Mixed distribution (linear and cubes)
    let mixed_values_2 = (1..=30).flat_map(|x| vec![x, x * x * x]).collect::<Vec<i32>>();
    for _ in 0..3 {
        let target = mixed_values_2[rng.gen_range(0..mixed_values_2.len())];
        let expected = mixed_values_2.binary_search(&target).ok();
        test_data.push((mixed_values_2.clone(), target, expected, "Mixed distribution (linear and cubes)".to_string()));
    }

    // Set 13: Mixed distribution (linear and powers of 2)
    let mixed_values_3 = (1..=20).flat_map(|x| vec![x, 2_i32.pow(x as u32)]).collect::<Vec<i32>>();
    for _ in 0..3 {
        let target = mixed_values_3[rng.gen_range(0..mixed_values_3.len())];
        let expected = mixed_values_3.binary_search(&target).ok();
        test_data.push((mixed_values_3.clone(), target, expected, "Mixed distribution (linear and powers of 2)".to_string()));
    }

    // Set 14: Mixed distribution (linear and factorials)
    let mixed_values_4 = (1..=10).flat_map(|x| vec![x, (1..=x).product()]).collect::<Vec<i32>>();
    for _ in 0..3 {
        let target = mixed_values_4[rng.gen_range(0..mixed_values_4.len())];
        let expected = mixed_values_4.binary_search(&target).ok();
        test_data.push((mixed_values_4.clone(), target, expected, "Mixed distribution (linear and factorials)".to_string()));
    }
    // Set 15: Mixed distribution (linear and Fibonacci sequence)
    let mixed_values_5 = {
        let mut fib = vec![0, 1];
        for i in 2..=20 {
            let next = fib[i - 1] + fib[i - 2];
            fib.push(next as i32);
        }
        (1..=20).flat_map(|x| vec![x as i32, fib[x]]).collect::<Vec<i32>>()
    };
    for _ in 0..3 {
        let target = mixed_values_5[rng.gen_range(0..mixed_values_5.len())];
        let expected = mixed_values_5.binary_search(&target).ok();
        test_data.push((mixed_values_5.clone(), target, expected, "Mixed distribution (linear and Fibonacci sequence)".to_string()));
    }

    // Set 16: Difficult distribution (prime numbers)
    let difficult_values_1 = (2..=100).filter(|x| (2..*x).all(|y| x % y != 0)).collect::<Vec<i32>>();
    for _ in 0..3 {
        let target = difficult_values_1[rng.gen_range(0..difficult_values_1.len())];
        let expected = difficult_values_1.binary_search(&target).ok();
        test_data.push((difficult_values_1.clone(), target, expected, "Difficult distribution (prime numbers)".to_string()));
    }

    // Set 17: Difficult distribution (random values)
    let difficult_values_2 = (0..100).map(|_| rng.gen_range(0..1000)).collect::<Vec<i32>>();
    for _ in 0..3 {
        let target = difficult_values_2[rng.gen_range(0..difficult_values_2.len())];
        let expected = difficult_values_2.binary_search(&target).ok();
        test_data.push((difficult_values_2.clone(), target, expected, "Difficult distribution (random values)".to_string()));
    }

    // Set 18: Difficult distribution (geometric progression)
    let difficult_values_3 = (0..20).map(|x| 2_i32.pow(x)).collect::<Vec<i32>>();
    for _ in 0..3 {
        let target = difficult_values_3[rng.gen_range(0..difficult_values_3.len())];
        let expected = difficult_values_3.binary_search(&target).ok();
        test_data.push((difficult_values_3.clone(), target, expected, "Difficult distribution (geometric progression)".to_string()));
    }

    // Set 19: Difficult distribution (exponential growth)
    let difficult_values_4 = (0..20).map(|x| (2.0_f64.powf(x as f64) as i32)).collect::<Vec<i32>>();
    for _ in 0..3 {
        let target = difficult_values_4[rng.gen_range(0..difficult_values_4.len())];
        let expected = difficult_values_4.binary_search(&target).ok();
        test_data.push((difficult_values_4.clone(), target, expected, "Difficult distribution (exponential growth)".to_string()));
    }

    // Set 20: Difficult distribution (logarithmic growth)
    let difficult_values_5 = (1..=20).map(|x| (x as f64).ln() as i32).collect::<Vec<i32>>();
    for _ in 0..3 {
        let target = difficult_values_5[rng.gen_range(0..difficult_values_5.len())];
        let expected = difficult_values_5.binary_search(&target).ok();
        test_data.push((difficult_values_5.clone(), target, expected, "Difficult distribution (logarithmic growth)".to_string()));
    }

    // Set 21: Difficult distribution (alternating high-low values)
    let difficult_values_6 = (0..100).map(|x| if x % 2 == 0 { x * 10 } else { x }).collect::<Vec<i32>>();
    for _ in 0..3 {
        let target = difficult_values_6[rng.gen_range(0..difficult_values_6.len())];
        let expected = difficult_values_6.binary_search(&target).ok();
        test_data.push((difficult_values_6.clone(), target, expected, "Difficult distribution (alternating high-low values)".to_string()));
    }

    // Set 22: Difficult distribution (repeated blocks)
    let difficult_values_7 = (0..10).flat_map(|x| vec![x; 10]).collect::<Vec<i32>>();
    for _ in 0..3 {
        let target = difficult_values_7[rng.gen_range(0..difficult_values_7.len())];
        let expected = difficult_values_7.binary_search(&target).ok();
        test_data.push((difficult_values_7.clone(), target, expected, "Difficult distribution (repeated blocks)".to_string()));
    }

    // Set 23: Difficult distribution (sparse values)
    let difficult_values_8 = (0..20).map(|x| x * 1000).collect::<Vec<i32>>();
    for _ in 0..3 {
        let target = difficult_values_8[rng.gen_range(0..difficult_values_8.len())];
        let expected = difficult_values_8.binary_search(&target).ok();
        test_data.push((difficult_values_8.clone(), target, expected, "Difficult distribution (sparse values)".to_string()));
    }

    // Set 24: Difficult distribution (negative and positive values)
    let difficult_values_9 = (-50..50).collect::<Vec<i32>>();
    for _ in 0..3 {
        let target = difficult_values_9[rng.gen_range(0..difficult_values_9.len())];
        let expected = difficult_values_9.binary_search(&target).ok();
        test_data.push((difficult_values_9.clone(), target, expected, "Difficult distribution (negative and positive values)".to_string()));
    }

    // Set 25: Difficult distribution (random large values)
    let difficult_values_10 = (0..100).map(|_| rng.gen_range(10000..100000)).collect::<Vec<i32>>();
    for _ in 0..3 {
        let target = difficult_values_10[rng.gen_range(0..difficult_values_10.len())];
        let expected = difficult_values_10.binary_search(&target).ok();
        test_data.push((difficult_values_10.clone(), target, expected, "Difficult distribution (random large values)".to_string()));
    }

    // Set 26: Difficult distribution (prime numbers)
    let difficult_values_11 = vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97];
    for _ in 0..3 {
        let target = difficult_values_11[rng.gen_range(0..difficult_values_11.len())];
        let expected = difficult_values_11.binary_search(&target).ok();
        test_data.push((difficult_values_11.clone(), target, expected, "Difficult distribution (prime numbers)".to_string()));
    }

    // Set 27: Difficult distribution (fibonacci sequence)
    let difficult_values_12 = vec![1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765];
    for _ in 0..3 {
        let target = difficult_values_12[rng.gen_range(0..difficult_values_12.len())];
        let expected = difficult_values_12.binary_search(&target).ok();
        test_data.push((difficult_values_12.clone(), target, expected, "Difficult distribution (fibonacci sequence)".to_string()));
    }

    // Set 28: Difficult distribution (geometric progression)
    let difficult_values_13 = (0..20).map(|x| 2_i32.pow(x)).collect::<Vec<i32>>();
    for _ in 0..3 {
        let target = difficult_values_13[rng.gen_range(0..difficult_values_13.len())];
        let expected = difficult_values_13.binary_search(&target).ok();
        test_data.push((difficult_values_13.clone(), target, expected, "Difficult distribution (geometric progression)".to_string()));
    }

    // Set 29: Difficult distribution (random gaps)
    let mut difficult_values_14 = vec![0];
    for _ in 0..19 {
        difficult_values_14.push(difficult_values_14.last().unwrap() + rng.gen_range(1..100));
    }
    for _ in 0..3 {
        let target = difficult_values_14[rng.gen_range(0..difficult_values_14.len())];
        let expected = difficult_values_14.binary_search(&target).ok();
        test_data.push((difficult_values_14.clone(), target, expected, "Difficult distribution (random gaps)".to_string()));
    }

    // Set 30: Difficult distribution (exponential growth)
    let difficult_values_15 = (0..20).map(|x| (2.0_f64.powi(x) as i32)).collect::<Vec<i32>>();
    for _ in 0..3 {
        let target = difficult_values_15[rng.gen_range(0..difficult_values_15.len())];
        let expected = difficult_values_15.binary_search(&target).ok();
        test_data.push((difficult_values_15.clone(), target, expected, "Difficult distribution (exponential growth)".to_string()));
    }

    test_data
}

fn main() {
    let test_data = generate_test_data();
    let mut iteration = 0;

    // Initialize win counters
    let mut linear_fit_wins = 0;
    let mut binary_search_wins = 0;
    let mut interpolated_binary_search_wins = 0;
    let mut hybrid_search_wins = 0;
    let mut hybrid_interpolated_search_wins = 0;
    let mut hybrid_adaptive_search_wins = 0;

    for (values, search_value, expected, comment) in test_data {
        println!("{}", comment.bold().blue());
        iteration += 1;
        let num_elements = values.len();

        // Linear Fit Search
        let start = Instant::now();
        let result = linear_fit_search(&values, search_value);
        let duration = start.elapsed();
        let memory_usage = std::mem::size_of_val(&values) + std::mem::size_of_val(&search_value) + std::mem::size_of_val(&result);
        {
            let mut cpu_metrics = CPU_METRICS.lock().unwrap();
            cpu_metrics.entry("linear_fit_search").or_default().push(duration.as_nanos());
            let mut memory_metrics = MEMORY_METRICS.lock().unwrap();
            memory_metrics.entry("linear_fit_search").or_default().push(memory_usage);
        }
        let normalized_linear_duration = duration.as_nanos() as f64 / num_elements as f64;
        println!(
            "{}\n{}:\nvalue = {}\nexpected = {:?}\nresult = {:?}\nduration = {:?}\nnormalized duration per element = {:.2} ns\nmemory usage = {} bytes",
            format!("Test Case #{}", iteration).bold().yellow(),
            "Linear Fit Search".bold().green(),
            search_value, expected, result, duration, normalized_linear_duration, memory_usage
        );

        // Binary Search
        let start = Instant::now();
        let result = binary_search(&values, search_value);
        let duration = start.elapsed();
        let memory_usage = std::mem::size_of_val(&values) + std::mem::size_of_val(&search_value) + std::mem::size_of_val(&result);
        {
            let mut cpu_metrics = CPU_METRICS.lock().unwrap();
            cpu_metrics.entry("binary_search").or_default().push(duration.as_nanos());
            let mut memory_metrics = MEMORY_METRICS.lock().unwrap();
            memory_metrics.entry("binary_search").or_default().push(memory_usage);
        }
        let normalized_binary_duration = duration.as_nanos() as f64 / num_elements as f64;
        println!(
            "{}\n{}:\nvalue = {}\nexpected = {:?}\nresult = {:?}\nduration = {:?}\nnormalized duration per element = {:.2} ns\nmemory usage = {} bytes",
            format!("Test Case #{}", iteration).bold().yellow(),
            "Binary Search".bold().green(),
            search_value, expected, result, duration, normalized_binary_duration, memory_usage
        );

        // Interpolated Binary Search
        let start = Instant::now();
        let result = interpolated_binary_search(&values, search_value);
        let duration = start.elapsed();
        let memory_usage = std::mem::size_of_val(&values) + std::mem::size_of_val(&search_value) + std::mem::size_of_val(&result);
        {
            let mut cpu_metrics = CPU_METRICS.lock().unwrap();
            cpu_metrics.entry("interpolated_binary_search").or_default().push(duration.as_nanos());
            let mut memory_metrics = MEMORY_METRICS.lock().unwrap();
            memory_metrics.entry("interpolated_binary_search").or_default().push(memory_usage);
        }
        let normalized_interpolated_duration = duration.as_nanos() as f64 / num_elements as f64;
        println!(
            "{}\n{}:\nvalue = {}\nexpected = {:?}\nresult = {:?}\nduration = {:?}\nnormalized duration per element = {:.2} ns\nmemory usage = {} bytes",
            format!("Test Case #{}", iteration).bold().yellow(),
            "Interpolated Binary Search".bold().green(),
            search_value, expected, result, duration, normalized_interpolated_duration, memory_usage
        );

        // Hybrid Search
        let start = Instant::now();
        let result = hybrid_search(&values, search_value);
        let duration = start.elapsed();
        let memory_usage = std::mem::size_of_val(&values) + std::mem::size_of_val(&search_value) + std::mem::size_of_val(&result);
        {
            let mut cpu_metrics = CPU_METRICS.lock().unwrap();
            cpu_metrics.entry("hybrid_search").or_default().push(duration.as_nanos());
            let mut memory_metrics = MEMORY_METRICS.lock().unwrap();
            memory_metrics.entry("hybrid_search").or_default().push(memory_usage);
        }
        let normalized_hybrid_duration = duration.as_nanos() as f64 / num_elements as f64;
        println!(
            "{}\n{}:\nvalue = {}\nexpected = {:?}\nresult = {:?}\nduration = {:?}\nnormalized duration per element = {:.2} ns\nmemory usage = {} bytes",
            format!("Test Case #{}", iteration).bold().yellow(),
            "Hybrid Linear-Binary Optimized Threshold Search".bold().green(),
            search_value, expected, result, duration, normalized_hybrid_duration, memory_usage
        );

        // Hybrid Interpolated Search
        let start = Instant::now();
        let result = hybrid_interpolated_search(&values, search_value);
        let duration = start.elapsed();
        let memory_usage = std::mem::size_of_val(&values) + std::mem::size_of_val(&search_value) + std::mem::size_of_val(&result);
        {
            let mut cpu_metrics = CPU_METRICS.lock().unwrap();
            cpu_metrics.entry("hybrid_interpolated_search").or_default().push(duration.as_nanos());
            let mut memory_metrics = MEMORY_METRICS.lock().unwrap();
            memory_metrics.entry("hybrid_interpolated_search").or_default().push(memory_usage);
        }
        let normalized_hybrid_interpolated_duration = duration.as_nanos() as f64 / num_elements as f64;
        println!(
            "{}\n{}:\nvalue = {}\nexpected = {:?}\nresult = {:?}\nduration = {:?}\nnormalized duration per element = {:.2} ns\nmemory usage = {} bytes",
            format!("Test Case #{}", iteration).bold().yellow(),
            "Hybrid Linear-Interpolated Binary Search".bold().green(),
            search_value, expected, result, duration, normalized_hybrid_interpolated_duration, memory_usage
        );

        // Hybrid Adaptive Search
        let start = Instant::now();
        let result = hybrid_adaptive_search(&values, search_value);
        let duration = start.elapsed();
        let memory_usage = std::mem::size_of_val(&values) + std::mem::size_of_val(&search_value) + std::mem::size_of_val(&result);
        {
            let mut cpu_metrics = CPU_METRICS.lock().unwrap();
            cpu_metrics.entry("hybrid_adaptive_search").or_default().push(duration.as_nanos());
            let mut memory_metrics = MEMORY_METRICS.lock().unwrap();
            memory_metrics.entry("hybrid_adaptive_search").or_default().push(memory_usage);
        }
        let normalized_hybrid_adaptive_duration = duration.as_nanos() as f64 / num_elements as f64;
        println!(
            "{}\n{}:\nvalue = {}\nexpected = {:?}\nresult = {:?}\nduration = {:?}\nnormalized duration per element = {:.2} ns\nmemory usage = {} bytes",
            format!("Test Case #{}", iteration).bold().yellow(),
            "Hybrid Linear-Adaptive Binary Search".bold().green(),
            search_value, expected, result, duration, normalized_hybrid_adaptive_duration, memory_usage
        );

        // Bind the array to a variable to extend its lifetime
        let search_methods = [
            ("Linear Fit Search", normalized_linear_duration),
            ("Binary Search", normalized_binary_duration),
            ("Interpolated Binary Search", normalized_interpolated_duration),
            ("Hybrid Linear-Binary Optimized Threshold Search", normalized_hybrid_duration),
            ("Hybrid Linear-Interpolated Binary Search", normalized_hybrid_interpolated_duration),
            ("Hybrid Linear-Adaptive Binary Search", normalized_hybrid_adaptive_duration),
        ];

        // Determine the winner based on the least runtime
        let (winner, _) = search_methods
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        let winner_str = winner.to_string();
        match winner_str.as_str() {
            "Linear Fit Search" => linear_fit_wins += 1,
            "Binary Search" => binary_search_wins += 1,
            "Interpolated Binary Search" => interpolated_binary_search_wins += 1,
            "Hybrid Linear-Binary Optimized Threshold Search" => hybrid_search_wins += 1,
            "Hybrid Linear-Interpolated Binary Search" => hybrid_interpolated_search_wins += 1,
            "Hybrid Linear-Adaptive Binary Search" => hybrid_adaptive_search_wins += 1,
            _ => (),
        }
        println!("{}", format!("WINNER: {}", winner_str).bold().blue());

    }

    // Print total wins
    println!("\n\x1b[31mTotal Wins:\x1b[0m");
    println!("Linear Fit Search: {}", linear_fit_wins);
    println!("Binary Search: {}", binary_search_wins);
    println!("Interpolated Binary Search: {}", interpolated_binary_search_wins);
    println!("Hybrid Linear-Binary Optimized Threshold Search: {}", hybrid_search_wins);
    println!("Hybrid Linear-Interpolated Binary Search: {}", hybrid_interpolated_search_wins);
    println!("Hybrid Linear-Adaptive Binary Search: {}", hybrid_adaptive_search_wins);
}