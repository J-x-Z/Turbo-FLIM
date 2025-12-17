use ndarray::{Array1, Axis};
use rand::thread_rng;
use rand_distr::{Distribution, Poisson};
use std::f64::consts::PI;

pub fn generate_time_axis(dt: f64, max_t: f64) -> Array1<f64> {
    Array1::range(0.0, max_t + dt, dt)
}

pub fn generate_irf(t: &Array1<f64>, sigma: f64, center: f64) -> Array1<f64> {
    let mut irf = t.mapv(|val| {
        (-((val - center).powi(2)) / (2.0 * sigma.powi(2))).exp()
    });
    
    // Normalize to sum = 1 (approximation of integral = 1 if divided by dt, but usually we normalize sum)
    let sum = irf.sum();
    if sum > 0.0 {
        irf /= sum;
    }
    irf
}

pub fn generate_decay(t: &Array1<f64>, taus: &[f64], amps: &[f64]) -> Array1<f64> {
    let mut decay = Array1::zeros(t.len());
    for (i, &tau) in taus.iter().enumerate() {
        let amp = amps.get(i).cloned().unwrap_or(1.0);
        decay = decay + t.mapv(|val| amp * (-val / tau).exp());
    }
    decay
}

pub fn add_poisson_noise(signal: &Array1<f64>, peak_counts: f64) -> Array1<f64> {
    let max_val = signal.fold(0.0f64, |a, &b| a.max(b));
    let scale = if max_val > 0.0 { peak_counts / max_val } else { 0.0 };
    
    let scaled_signal = signal * scale;
    let mut rng = thread_rng();
    
    scaled_signal.mapv(|val| {
        if val > 0.0 {
            let poi = Poisson::new(val).unwrap();
            poi.sample(&mut rng)
        } else {
            0.0
        }
    })
}
