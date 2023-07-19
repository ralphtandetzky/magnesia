use std::{
    array::from_fn,
    ops::{Add, Mul, Sub},
};

/// Data structure implementing the differential evolution algorithm.
///
/// # Example
/// ```
/// use magnesia::optimize::test_functions::ackley;
/// use magnesia::optimize::differential_evolution;
///
/// let mut rng = rand::thread_rng();
/// let loc = differential_evolution(
///     &[(-5.0, 5.0); 2],
///     20,
///     0.9,
///     0.8,
///     500,
///     &mut rng,
///     |lhs, rhs| ackley(lhs[0], lhs[1]) < ackley(rhs[0], rhs[1]),
/// );
/// assert!(loc[0].abs() < 0.1);
/// assert!(loc[1].abs() < 0.1);
/// ```
pub fn optimize<T, const N: usize>(
    bounds: &[(T, T); N],
    population_size: usize,
    crossover_probability: f32,
    differential_weight: T,
    num_iters: usize,
    rng: &mut impl rand::Rng,
    mut compare_candidates: impl FnMut(&[T; N], &[T; N]) -> bool,
) -> [T; N]
where
    T: PartialOrd
        + Clone
        + Copy
        + Add<T, Output = T>
        + Sub<T, Output = T>
        + Mul<T, Output = T>
        + rand::distributions::uniform::SampleUniform,
    f32: Into<T>,
    i8: Into<T>,
{
    assert!(
        population_size >= 4,
        "The population must have at least 4 elements"
    );
    assert!(
        (0.0..=1.0).contains(&crossover_probability),
        "Invalid crossover probability"
    );
    assert!(
        (0.into()..=2.into()).contains(&differential_weight),
        "Invalid differential weight"
    );

    // Create initial candidatre set
    let mut population = (0..population_size)
        .map(|_| from_fn(|i| rng.gen_range(bounds[i].0..bounds[i].1)))
        .collect::<Box<[[T; N]]>>();

    // Greedily improve solution
    let index_range = 0..population.len();
    for i in index_range.clone().cycle().take(num_iters) {
        // Generate random distinct indices `j`, `k` and `l`
        let mut j;
        loop {
            j = rng.gen_range(index_range.clone());
            if j != i {
                break;
            }
        }
        let mut k;
        loop {
            k = rng.gen_range(index_range.clone());
            if ![i, j].contains(&k) {
                break;
            }
        }
        let mut l;
        loop {
            l = rng.gen_range(index_range.clone());
            if ![i, j, k].contains(&l) {
                break;
            }
        }
        // Freeze j, k and l
        let (j, k, l) = (j, k, l);

        // Generate new candidate
        let r = rng.gen_range(0..N);
        let x = from_fn(|n| {
            if n == r || rng.gen::<f32>() < crossover_probability {
                population[j][n] + (population[k][n] - population[l][n]) * differential_weight
            } else {
                population[i][n]
            }
        });

        // If the candidate is bettern than population[i], replace
        if compare_candidates(&x, &population[i]) {
            population[i] = x;
        }
    }

    // knockout stage
    let mut population = population.as_mut();
    while population.len() > 1 {
        let pop_len = population.len();
        let half_len = pop_len / 2;
        for i in 0..half_len {
            if compare_candidates(&population[half_len + i], &population[i]) {
                population[i] = population[half_len + i];
            }
            if pop_len % 2 == 1 {
                population[half_len] = population[pop_len - 1];
            }
        }
        population = &mut population[0..((pop_len + 1) / 2)];
    }

    // Return the winner
    population[0]
}

#[test]
fn test_optimize_ackley() {
    use crate::optimize::test_functions::ackley;

    let mut rng = rand::thread_rng();
    for _ in 0..10 {
        let loc = optimize(
            &[(-5.0, 5.0); 2],
            20,
            0.9,
            0.8,
            500,
            &mut rng,
            |lhs, rhs| ackley(lhs[0], lhs[1]) < ackley(rhs[0], rhs[1]),
        );
        assert!(loc[0].abs() < 0.1);
        assert!(loc[1].abs() < 0.1);
    }
}
