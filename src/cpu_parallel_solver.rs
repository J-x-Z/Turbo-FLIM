use rayon::prelude::*;
use ndarray::Array1;
use crate::solver::solve_phase_plane;

pub fn run_cpu_parallel(
    t: &Array1<f64>,
    irf: &Array1<f64>,
    signals: &Vec<Array1<f64>>,
    dt: f64,
) -> Vec<(f64, f64)> {
    // Parallel iterator over signals
    let results: Vec<(f64, f64)> = signals.par_iter().map(|signal| {
        let roots = solve_phase_plane(t, signal, irf, dt, 2);
        if roots.len() == 2 {
            (roots[0], roots[1])
        } else if roots.len() == 1 {
            (roots[0], 0.0)
        } else {
            (0.0, 0.0)
        }
    }).collect();
    
    results
}
