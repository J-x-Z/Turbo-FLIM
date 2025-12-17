use ndarray::Array1;

/// Performs convolution of signal and IRF with time step dt.
/// Replicates the logic of Convolution.m: integral based convolution.
pub fn convolution(signal: &Array1<f64>, irf: &Array1<f64>, dt: f64) -> Array1<f64> {
    let n = signal.len();
    let mut result = Array1::zeros(n);

    // Naive convolution O(N^2) matching the Matlab implementation logic
    // integral_{0}^{t} f(t') g(t-t') dt'
    // discrete: sum(f[j] * g[i-j] * dt)
    
    // Note: Matlab's conv is usually full convolution. The user's Convolution.m
    // was a custom integral implementation. We should stick to standard discrete convolution 
    // scaled by dt to match the physical meaning of integral.
    
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..=i {
            // signal[j] * irf[i-j] * dt
            sum += signal[j] * irf[i - j];
        }
        result[i] = sum * dt;
    }
    
    result
}
