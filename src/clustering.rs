use ndarray::{Array1, Array2, Axis};
use std::f64::consts::PI;

/// Simple GMM with 2 components (Diagonal Covariance for stability)
pub struct GMM {
    pub means: Vec<[f64; 2]>,
    pub variances: Vec<[f64; 2]>,
    pub weights: Vec<f64>,
}

impl GMM {
    pub fn new() -> Self {
        GMM {
            means: vec![[0.0; 2]; 2],
            variances: vec![[1.0; 2]; 2],
            weights: vec![0.5, 0.5],
        }
    }

    fn gaussian_pdf(&self, x: &[f64; 2], mean: &[f64; 2], var: &[f64; 2]) -> f64 {
        let det = var[0] * var[1];
        let norm = 1.0 / (2.0 * PI * det.sqrt());
        let diff0 = x[0] - mean[0];
        let diff1 = x[1] - mean[1];
        let exp_term = -0.5 * (diff0 * diff0 / var[0] + diff1 * diff1 / var[1]);
        norm * exp_term.exp()
    }

    /// Train using Expectation-Maximization (EM) Algorithm
    pub fn fit(&mut self, data: &Vec<[f64; 2]>, steps: usize) {
        let n = data.len();
        if n == 0 { return; }
        
        // Initialization (K-Means++ style or simple split)
        // Sort by T1 to separate initially
        let mut sorted_data = data.clone();
        sorted_data.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());
        
        self.means[0] = sorted_data[n / 4];         // Lower T1
        self.means[1] = sorted_data[3 * n / 4];     // Higher T1
        
        let mut responsibilities = vec![[0.0; 2]; n];
        
        for _step in 0..steps {
            // E-Step: Calculate responsibilities (probabilities)
            for (i, x) in data.iter().enumerate() {
                let p0 = self.weights[0] * self.gaussian_pdf(x, &self.means[0], &self.variances[0]);
                let p1 = self.weights[1] * self.gaussian_pdf(x, &self.means[1], &self.variances[1]);
                let total = p0 + p1;
                if total > 1e-12 {
                    responsibilities[i][0] = p0 / total;
                    responsibilities[i][1] = p1 / total;
                } else {
                    responsibilities[i] = [0.5, 0.5];
                }
            }
            
            // M-Step: Update parameters
            let mut sum_resp = [0.0; 2];
            let mut new_means = [[0.0; 2]; 2];
            let mut new_vars = [[0.0; 2]; 2];
            
            for (i, x) in data.iter().enumerate() {
                for k in 0..2 {
                    let r = responsibilities[i][k];
                    sum_resp[k] += r;
                    new_means[k][0] += r * x[0];
                    new_means[k][1] += r * x[1];
                }
            }
            
            // Normalize Means
            for k in 0..2 {
                if sum_resp[k] > 0.0 {
                    self.means[k][0] = new_means[k][0] / sum_resp[k];
                    self.means[k][1] = new_means[k][1] / sum_resp[k];
                    self.weights[k] = sum_resp[k] / n as f64;
                }
            }
            
            // Update Variances
             for (i, x) in data.iter().enumerate() {
                for k in 0..2 {
                    let r = responsibilities[i][k];
                    let d0 = x[0] - self.means[k][0];
                    let d1 = x[1] - self.means[k][1];
                    new_vars[k][0] += r * d0 * d0;
                    new_vars[k][1] += r * d1 * d1;
                }
            }
            
            for k in 0..2 {
                 if sum_resp[k] > 0.0 {
                    self.variances[k][0] = new_vars[k][0] / sum_resp[k] + 1e-6; // Add epsilon stability
                    self.variances[k][1] = new_vars[k][1] / sum_resp[k] + 1e-6;
                 }
            }
        }
    }
    
    pub fn predict_proba(&self, x: &[f64; 2]) -> f64 {
        let p0 = self.weights[0] * self.gaussian_pdf(x, &self.means[0], &self.variances[0]);
        let p1 = self.weights[1] * self.gaussian_pdf(x, &self.means[1], &self.variances[1]);
        let total = p0 + p1;
        if total > 0.0 {
            p1 / total // Probability of being Class 1 (High T1/Tumor depending on init)
        } else {
            0.5
        }
    }
}

pub fn run_clustering(
    results: &Vec<(f64, f64)>,
) -> (Vec<usize>, Vec<f64>) {
    // 1. Data Prep
    let valid_data: Vec<[f64; 2]> = results.iter()
        .filter(|(t1, t2)| *t1 > 0.0 && *t2 > 0.0)
        .map(|(t1, t2)| [*t1, *t2])
        .collect();
    
    // 2. Train GMM
    let mut gmm = GMM::new();
    gmm.fit(&valid_data, 50); // 50 EM iterations
    
    // 3. Predict (Hard class + Probability)
    let mut assignments = Vec::new(); // 0 or 1
    let mut probs = Vec::new();       // Probability of class 1
    
    // Remap to original indices (filling invalid with 0)
    let mut valid_idx = 0;
    for (t1, t2) in results {
        if *t1 > 0.0 && *t2 > 0.0 {
            let p1 = gmm.predict_proba(&[*t1, *t2]);
            probs.push(p1);
            if p1 > 0.5 { assignments.push(1); } else { assignments.push(0); }
            valid_idx += 1;
        } else {
            assignments.push(0);
            probs.push(0.0);
        }
    }
    
    (assignments, probs)
}
