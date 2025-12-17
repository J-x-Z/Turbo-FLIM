use ndarray::Array1;
use nalgebra::{DMatrix, DVector};
use crate::utils::convolution;

pub fn solve_phase_plane(
    t: &Array1<f64>,
    signal: &Array1<f64>,
    irf: &Array1<f64>,
    dt: f64,
    n_components: usize,
) -> Vec<f64> {
    // 1. Normalization (Critical for numerical stability at high counts)
    let max_val = signal.fold(0.0f64, |a, &b| a.max(b.abs()));
    let signal_norm = if max_val > 0.0 { signal / max_val } else { signal.clone() };

    // 2. Compute F and G integrals (basis function convolutions)
    // F_k = conv(signal, t^(k-1)/(k-1)!)
    // G_k = conv(signal, t^k/k!)
    
    let n_points = t.len();
    let mut f_integrals = Vec::new(); // Stores entire array for each k
    let mut g_integrals = Vec::new(); 

    // Calculate basis functions and convolve
    for k in 1..=n_components {
        let k_idx = k as i32;
        
        // phi_k = t^(k-1) / (k-1)!
        // This is the SAME basis function used for both F and G terms in Apanasovich method
        let phi_k = t.mapv(|val| val.powi(k_idx - 1) / factorial(k_idx - 1));
        
        // F_k corresponds to Signal convolved with Basis
        // F(k, :) = Convolution(fi, phi, dt)
        let f_conv = convolution(&signal_norm, &phi_k, dt);
        f_integrals.push(f_conv);

        // G_k corresponds to IRF convolved with Basis
        // G(k, :) = Convolution(gi, phi, dt)  <-- gi is IRF
        let g_conv = convolution(irf, &phi_k, dt);
        g_integrals.push(g_conv);
    }
    
    // 3. Construct Linear System: A * C = B
    // A_matrix = [F*F', -F*G';
    //             -G*F',  G*G']
    // B_vector = [-F*fi; 
    //              G*fi]
    
    let num_t = t.len();
    let mut f_mat = DMatrix::zeros(n_components, num_t);
    let mut g_mat = DMatrix::zeros(n_components, num_t);
    
    for i in 0..n_components {
        for j in 0..num_t {
            f_mat[(i, j)] = f_integrals[i][j];
            g_mat[(i, j)] = g_integrals[i][j];
        }
    }
    
    // Build LHS Matrix (A) and RHS Vector (B)
    // System: A * X = B
    // In Matlab for N=2:
    // A = [F*F', -F*G'; -G*F', G*G']
    // B = [-F*y'; G*y'] where y is signal.
    
    let y_vec = DVector::from_column_slice(signal_norm.as_slice().unwrap());
    
    let ff_t = &f_mat * &f_mat.transpose();
    let fg_t = &f_mat * &g_mat.transpose();
    let gf_t = &g_mat * &f_mat.transpose();
    let gg_t = &g_mat * &g_mat.transpose();
    
    // Construct A (2n x 2n)
    let size = 2 * n_components;
    let mut a_mat = DMatrix::zeros(size, size);
    
    // Copy blocks
    // Top-Left: F*F'
    for r in 0..n_components {
        for c in 0..n_components {
            a_mat[(r, c)] = ff_t[(r, c)];
        }
    }
    // Top-Right: -F*G'
    for r in 0..n_components {
        for c in 0..n_components {
            a_mat[(r, c + n_components)] = -fg_t[(r, c)];
        }
    }
    // Bottom-Left: -G*F'
    for r in 0..n_components {
        for c in 0..n_components {
            a_mat[(r + n_components, c)] = -gf_t[(r, c)];
        }
    }
    // Bottom-Right: G*G'
    for r in 0..n_components {
        for c in 0..n_components {
            a_mat[(r + n_components, c + n_components)] = gg_t[(r, c)];
        }
    }
    
    // Construct B (2n x 1)
    let fy = &f_mat * &y_vec;
    let gy = &g_mat * &y_vec;
    
    let mut b_vec = DVector::zeros(size);
    for r in 0..n_components {
        b_vec[r] = -fy[r];
        b_vec[r + n_components] = gy[r];
    }
    
    // Solve
    match a_mat.try_inverse() {
        Some(inv) => {
            let x = inv * b_vec;
            // X contains coefficients of polynomial.
            // For N=1: X = [c1, c0] ?? No.
            // In Matlab: C_list = A \ B.
            // Then coeffs are constructed from C_list.
            // Matlab roots logic needs to be replicated.
            
            // Extract coefficients for characteristic polynomial
            // P(lambda) = c_n * lambda^n + ...
            // This part requires mapping X back to poly coeffs.
            // For N=1: X has 2 elements. c0 = x[0], c1 = x[1]?
            // Let's assume N=1 and N=2 specific logic for roots.
            
            find_roots(&x, n_components)
        },
        None => vec![], // Singular
    }
}

fn factorial(n: i32) -> f64 {
    (1..=n).fold(1.0, |acc, x| acc * x as f64)
}

fn find_roots(coeffs_x: &DVector<f64>, n: usize) -> Vec<f64> {
    // Logic derived from Matlab:
    // X contains coefficients [x0, x1, ... x_{n-1}] (first n elements of solution)
    // Coeffs array C constructed as [1, -x0, +x1, -x2 ...]
    // Then flipped: [..., +x1, -x0, 1]
    // Then roots found.
    
    // Extract first n elements
    let x = coeffs_x.rows(0, n);
    
    if n == 1 {
        // Poly: [-x0, 1] -> -x0 * tau + 1 = 0 -> tau = 1/x0
        let x0 = x[0];
        if x0.abs() > 1e-9 {
            vec![1.0 / x0]
        } else {
            vec![]
        }
    } else if n == 2 {
        // Poly: [x1, -x0, 1] -> x1 * tau^2 - x0 * tau + 1 = 0
        let x0 = x[0];
        let x1 = x[1];
        
        // Quadratic formula: ax^2 + bx + c = 0
        // a = x1, b = -x0, c = 1
        
        if x1.abs() < 1e-9 {
            // Degenerate to linear: -x0*tau + 1 = 0
            if x0.abs() > 1e-9 {
                return vec![1.0 / x0];
            } else {
                return vec![];
            }
        }
        
        let delta = x0 * x0 - 4.0 * x1 * 1.0;
        
        if delta < 0.0 {
            // Complex roots (physically invalid for lifetimes usually, or just noise)
            vec![]
        } else {
            let sqrt_delta = delta.sqrt();
            let t1 = (x0 - sqrt_delta) / (2.0 * x1);
            let t2 = (x0 + sqrt_delta) / (2.0 * x1);
            
            // Filter positive roots
            let mut roots = Vec::new();
            if t1 > 0.0 { roots.push(t1); }
            if t2 > 0.0 { roots.push(t2); }
            roots.sort_by(|a, b| a.partial_cmp(b).unwrap());
            roots
        }
    } else {
        // For N > 2, need general polynomial solver (e.g. eigenvalue of companion matrix)
        // Not implemented for this stage.
        vec![]
    }
}
