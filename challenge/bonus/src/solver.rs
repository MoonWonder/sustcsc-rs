use std::ops::{Add, Sub, Mul, Div, Neg};
use ndarray::{Array1, Array2, Axis, concatenate, Ix1};
use pyo3::prelude::*;
use pyo3::types::PyList;

/// Solves the LWE problem.
///
/// # Arguments
///
/// * `n` - Secret dimension
/// * `m` - Number of samples
/// * `q` - Modulus
/// * `alpha` - Relative error size
/// * `A` - Matrix of dimensions m x n (mod q)
/// * `b` - Vector of length m (mod q)
///
/// # Returns
///
/// * `Array1<u64>` - Recovered secret vector s of length n
pub(crate) fn solve_lwe(
    n: usize,
    m: usize,
    q: u64,
    alpha: f64,
    A: &Array2<u64>,
    b: &Array1<u64>,
) -> Array1<u64> {
    let pIm = Array2::from_shape_fn((m, m), |(i, j)| q as f64 * (i == j) as u64 as f64);
    let A = A.mapv(|x| x as f64);
    let M = concatenate(Axis(1), &[A.view(),pIm.view()]).unwrap();
    //let M = concatenate(Axis(1), &[pIm.view(),A.view()]).unwrap();
    //let pIn = Array2::from_shape_fn((n, n+m), |(i, j)| 1 as f64 * (i == j) as u64 as f64);
    //let N = concatenate(Axis(0), &[M.view(),pIn.view()]).unwrap();
    let b = b.mapv(|x| x as f64);
    let br = babai_nearest_vector(&M, &b);

    br.into_iter()
        .map(|x| (x.round() as i64).rem_euclid(q.try_into().unwrap()) as u64)
        .collect::<Array1<u64>>()
}

fn babai_nearest_vector(B: &Array2<f64>, t: &Array1<f64>) -> Array1<f64> {
    let mut A = B.clone().reversed_axes().to_owned();
    //for i in (0..A.nrows()) {
    //    println!("{}",A.row(i));
    //}
    lll(&mut A);
    let C = Gram_schmidt(&A);
    let mut b = Array1::zeros( A.ncols() );
    for i in (0..t.len()) {
        b[i] = t[i];
    }

    //for i in (0..A.nrows()) {
    //    println!("{}",A.row(i));
    //}

    for i in (0..A.nrows()).rev() { if C.row(i).dot(&C.row(i)) > 1e-6 {
        // Calculate coefficient: round to nearest integer
        let coeff = (b.dot(&C.row(i)) / C.row(i).dot(&C.row(i))).round();
        // Subtract the projection and store result back in b
        b = b - coeff * &A.row(i);
    } }

    //println!( "{}", b );
    //println!( "{}", t );
    let mut c = Array1::zeros(t.len());
    for i in (0..t.len()) {
        c[i] = t[i] - b[i];
    }
    //println!( "{}", c );
    c
}

fn Gram_schmidt( basis_t: &Array2<f64> ) -> Array2<f64> {
    let mut basis = basis_t.clone();
    let n = basis.nrows();
    for i in 0..n {
        let mut row_i = basis.row(i).to_owned();
        for j in 0..i { 
            let row_j  = basis.row(j);
            if row_j.dot(&row_j) > 1e-6 {
                let lambda = row_i.dot(&row_j) / row_j.dot(&row_j);
                row_i = row_i - lambda * &row_j;
            }
        }
        basis.row_mut(i).assign(&row_i);
    }
    basis
}

fn lll ( basis: &mut Array2<f64> ) -> PyResult<()> {
    Python::with_gil(|py| {
        let n = basis.nrows();
        let m = basis.ncols();

        let fpylll = py.import("fpylll")?;
        let integer_matrix = fpylll.getattr("IntegerMatrix")?;
        
        let matrix = integer_matrix.call1((n, m))?;
        
        // 设置矩阵元素
        for i in 0..n {
            for j in 0..m {
                matrix.call_method1("__setitem__", ((i, j), basis[ [i,j] ] as i64 ))?;
            }
        }

        // 导入LLL类
        let bkz = fpylll.getattr("BKZ")?;
        let reduction = bkz.getattr("reduction")?;
        let param = bkz.getattr("Param")?;
        let param_instance = param.call1((20,))?;
        let lll_instance = reduction.call1((matrix,param_instance,))?;

        for i in 0..n {
            for j in 0..m {
                let element = lll_instance.get_item((i, j))?;
                let value: f64 = element.extract()?;
                basis[[i, j]] = value;
            }
        }

        Ok(())
    })
}