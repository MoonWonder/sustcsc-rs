#![allow(non_snake_case, unused)]
mod lwe;
mod solver;

use crate::lwe::generate_lwe_instance;
use ndarray::Array1;
use std::time::Instant;

fn main() {
    let test_cases = vec![
    //    (5, 20, 7, 0.005, 0), // basic test. remove this line in contest
        (10, 100, 97, 0.005, 2),
        (20, 400, 193, 0.005, 3),
        (30, 900, 389, 0.005, 5),
        (40, 1500, 769, 0.005, 7),
        (45, 1700, 12289, 0.005, 9),
        (50, 2500, 1543, 0.005, 11),
        (55, 3600, 6151, 0.005, 13),
    ];

    let mut score = 0;
    let mut total = 0;
    println!("    #     n      m       q   alpha   error norm   time (s)  pass");
    println!("----- ----- ------ ------- ------- ------------ ---------- -----");

    for (i, &(n, m, q, alpha, cur_score)) in test_cases.iter().enumerate() {
        let instance = generate_lwe_instance(n, m, q, alpha);
        let start = Instant::now();
        let s_pred = solver::solve_lwe(n, m, q, alpha, &instance.a, &instance.b);
        let duration = start.elapsed();
        let error_norm = compute_error_norm(&instance.s, Array1::zeros(n), q);

        //println!("{}", instance.s);

        let pass = if s_pred.len() == n {
            validate_solution(&instance.s, &s_pred, q)
        } else {
            validate_solution(&instance.a.dot(&instance.s), &s_pred, q)
        };
        score += if pass { cur_score } else { 0 };
        total += duration.as_secs_f64() as usize;

        println!(
            "{:5} {:5} {:6} {:7} {:7.3} {:12.4} {:10.4}  {}",
            (i as i64) - 1,
            n,
            m,
            q,
            alpha,
            error_norm,
            duration.as_secs_f64(),
            if pass { "PASS" } else { "FAIL" }
        );
    }
    println!("Total score: {}\ttime: {}", score, total);
}

fn validate_solution(s_true: &Array1<u64>, s_pred: &Array1<u64>, q: u64) -> bool {
    if s_true.len() != s_pred.len() {
        panic!("s_true and s_pred must have the same length");
    }
    //for i in (0..s_pred.len()) {
    //    print!( "{} ", s_pred[i].rem_euclid( (q as i64).try_into().unwrap() ) );
    //}
    //println!(" ");
    //for i in (0..s_pred.len()) {
    //    print!( "{} ", s_true[i].rem_euclid( (q as i64).try_into().unwrap() ) );
    //}
    s_true
        .iter()
        .zip(s_pred.iter())
        .all(|(&a, &b)| (a as i64 - b as i64).rem_euclid(q as i64) == 0)
}

fn compute_error_norm(s_true: &Array1<u64>, s_pred: Array1<u64>, q: u64) -> f64 {
    s_true
        .iter()
        .zip(s_pred.iter())
        .map(|(&a, &b)| {
            let diff = (a as i64 - b as i64).rem_euclid(q as i64);
            let min_diff = diff.min(q as i64 - diff);
            (min_diff * min_diff) as f64
        })
        .sum::<f64>()
        .sqrt()
}
