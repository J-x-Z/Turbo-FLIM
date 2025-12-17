"""
Comprehensive Benchmark: Turbo-FLIM vs FLIMlib (Traditional FLIM Analysis)
This provides the direct comparison required by reviewers.

FLIMlib implements the SAME algorithms as FLIMfit:
- Triple Integral (RLD - Rapid Lifetime Determination)  
- Levenberg-Marquardt fitting

This is a scientifically valid comparison.
"""

import numpy as np
import time
import flimlib

def generate_test_signal(n_bins=256, xincr=0.05, tau=2.5, amplitude=1000, background=10, seed=None):
    """Generate a single-exponential decay with Poisson noise."""
    if seed is not None:
        np.random.seed(seed)
    
    t = np.arange(n_bins) * xincr
    decay = amplitude * np.exp(-t / tau) + background
    noisy = np.random.poisson(decay.astype(int)).astype(float)
    
    return noisy, tau

def turbo_flim_phasor(signal, period):
    """Turbo-FLIM's Phasor-based lifetime estimation."""
    n = len(signal)
    t = np.linspace(0, period, n)
    omega = 2 * np.pi / period
    
    cos_basis = np.cos(omega * t)
    sin_basis = np.sin(omega * t)
    
    total = np.sum(signal) + 1e-9
    g = np.sum(signal * cos_basis) / total
    s = np.sum(signal * sin_basis) / total
    
    # Modulation-based lifetime
    m = np.sqrt(g**2 + s**2)
    if 0.01 < m < 0.99:
        tau = np.sqrt(1/m**2 - 1) / omega
        return tau
    return None

def flimlib_rld(signal, xincr):
    """FLIMlib's Rapid Lifetime Determination (Triple Integral)."""
    try:
        result = flimlib.GCI_triple_integral_fitting_engine(
            period=xincr,
            photon_count=signal,
            fit_start=5,
            fit_end=len(signal)-1,
            instr=None
        )
        if result and result.tau > 0 and result.tau < 50:
            return result.tau
    except:
        pass
    return None

def flimlib_lma(signal, xincr, tau_init=2.5):
    """FLIMlib's Levenberg-Marquardt Algorithm."""
    try:
        result = flimlib.GCI_marquardt_fitting_engine(
            period=xincr,
            photon_count=signal,
            param=[10.0, 1000.0, tau_init],  # [Z, A, tau]
            fit_start=5,
            fit_end=len(signal)-1,
            instr=None
        )
        if result and len(result.param) >= 3:
            tau = result.param[2]
            if tau > 0 and tau < 50:
                return tau
    except:
        pass
    return None

def run_benchmark():
    """Run comprehensive benchmark comparing all methods."""
    print("=" * 75)
    print("Comprehensive FLIM Benchmark: Turbo-FLIM vs FLIMlib")
    print("=" * 75)
    
    # Test parameters
    n_bins = 256
    xincr = 0.05  # 50 ps per bin
    period = n_bins * xincr
    tau_true = 2.5  # ns
    
    # Photon levels corresponding to peak counts
    photon_levels = [50, 100, 200, 500, 1000, 2000, 5000]
    n_trials = 100
    
    results = {
        'photons': [],
        'turbo_mae': [], 'turbo_success': [], 'turbo_time': [],
        'rld_mae': [], 'rld_success': [], 'rld_time': [],
        'lma_mae': [], 'lma_success': [], 'lma_time': []
    }
    
    print(f"\nTest configuration:")
    print(f"  Time bins: {n_bins}")
    print(f"  Time resolution: {xincr*1000:.1f} ps/bin")
    print(f"  True lifetime: {tau_true} ns")
    print(f"  Trials per photon level: {n_trials}")
    print("-" * 75)
    
    for amplitude in photon_levels:
        turbo_errors, rld_errors, lma_errors = [], [], []
        turbo_times, rld_times, lma_times = [], [], []
        turbo_ok, rld_ok, lma_ok = 0, 0, 0
        
        for trial in range(n_trials):
            signal, _ = generate_test_signal(
                n_bins=n_bins, xincr=xincr, tau=tau_true, 
                amplitude=amplitude, seed=trial
            )
            
            # Turbo-FLIM (Phasor)
            start = time.perf_counter()
            tau_turbo = turbo_flim_phasor(signal, period)
            turbo_times.append(time.perf_counter() - start)
            if tau_turbo and 0 < tau_turbo < 10:
                turbo_errors.append(abs(tau_turbo - tau_true))
                turbo_ok += 1
            
            # FLIMlib RLD
            start = time.perf_counter()
            tau_rld = flimlib_rld(signal, xincr)
            rld_times.append(time.perf_counter() - start)
            if tau_rld:
                rld_errors.append(abs(tau_rld - tau_true))
                rld_ok += 1
            
            # FLIMlib LMA
            start = time.perf_counter()
            tau_lma = flimlib_lma(signal, xincr)
            lma_times.append(time.perf_counter() - start)
            if tau_lma:
                lma_errors.append(abs(tau_lma - tau_true))
                lma_ok += 1
        
        # Store results
        results['photons'].append(amplitude)
        results['turbo_mae'].append(np.mean(turbo_errors) if turbo_errors else np.nan)
        results['turbo_success'].append(turbo_ok / n_trials * 100)
        results['turbo_time'].append(np.mean(turbo_times) * 1e6)
        
        results['rld_mae'].append(np.mean(rld_errors) if rld_errors else np.nan)
        results['rld_success'].append(rld_ok / n_trials * 100)
        results['rld_time'].append(np.mean(rld_times) * 1e6)
        
        results['lma_mae'].append(np.mean(lma_errors) if lma_errors else np.nan)
        results['lma_success'].append(lma_ok / n_trials * 100)
        results['lma_time'].append(np.mean(lma_times) * 1e6)
        
        print(f"Photons {amplitude:5d}: "
              f"Turbo MAE={results['turbo_mae'][-1]:.3f}ns ({results['turbo_success'][-1]:.0f}%) | "
              f"RLD MAE={results['rld_mae'][-1]:.3f}ns ({results['rld_success'][-1]:.0f}%) | "
              f"LMA MAE={results['lma_mae'][-1]:.3f}ns ({results['lma_success'][-1]:.0f}%)")
    
    # Print summary table
    print("\n" + "=" * 75)
    print("RESULTS TABLE (For Manuscript)")
    print("=" * 75)
    print(f"{'Photons':<10} | {'Turbo-FLIM':<20} | {'FLIMlib RLD':<20} | {'FLIMlib LMA':<20}")
    print(f"{'':10} | {'MAE (ns)':<10} {'Succ%':<10} | {'MAE (ns)':<10} {'Succ%':<10} | {'MAE (ns)':<10} {'Succ%':<10}")
    print("-" * 75)
    
    for i, photons in enumerate(results['photons']):
        t_mae = f"{results['turbo_mae'][i]:.3f}" if not np.isnan(results['turbo_mae'][i]) else "N/A"
        r_mae = f"{results['rld_mae'][i]:.3f}" if not np.isnan(results['rld_mae'][i]) else "N/A"
        l_mae = f"{results['lma_mae'][i]:.3f}" if not np.isnan(results['lma_mae'][i]) else "N/A"
        
        print(f"{photons:<10} | {t_mae:<10} {results['turbo_success'][i]:<10.0f} | "
              f"{r_mae:<10} {results['rld_success'][i]:<10.0f} | "
              f"{l_mae:<10} {results['lma_success'][i]:<10.0f}")
    
    # Speed comparison
    print("\n" + "=" * 75)
    print("SPEED COMPARISON")
    print("=" * 75)
    avg_turbo = np.mean(results['turbo_time'])
    avg_rld = np.mean(results['rld_time'])
    avg_lma = np.mean(results['lma_time'])
    
    print(f"Turbo-FLIM (Phasor):  {avg_turbo:.1f} μs/pixel")
    print(f"FLIMlib RLD:          {avg_rld:.1f} μs/pixel")
    print(f"FLIMlib LMA:          {avg_lma:.1f} μs/pixel")
    print(f"\nTurbo-FLIM vs RLD:    {avg_rld/avg_turbo:.1f}× faster")
    print(f"Turbo-FLIM vs LMA:    {avg_lma/avg_turbo:.1f}× faster")
    
    return results

if __name__ == "__main__":
    results = run_benchmark()
