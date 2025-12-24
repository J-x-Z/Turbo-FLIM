"""
Core analysis module for PyTurboFLIM.

Provides the main TurboFLIM class and convenience functions for
fluorescence lifetime analysis using physics-embedded neural networks.
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from .phasor import compute_phasor_batch


class TurboFLIM:
    """
    Physics-guided deep learning model for fluorescence lifetime analysis.
    
    The model uses a Phasor-Fusion architecture that combines raw decay
    histograms with Fourier-domain phasor coordinates to achieve robust
    lifetime estimation even in low-photon regimes.
    
    Attributes:
        model: Trained MLPRegressor model
        laser_period: Laser repetition period in nanoseconds
        is_fitted: Whether the model has been trained
    
    Example:
        >>> model = TurboFLIM(laser_period=12.5)
        >>> model.fit(X_train, y_train)
        >>> lifetimes = model.predict(X_test)
    """
    
    def __init__(self, laser_period=12.5, hidden_layers=(256, 128, 64, 32),
                 learning_rate=0.001, max_iter=500, random_state=42):
        """
        Initialize TurboFLIM model.
        
        Args:
            laser_period: Laser repetition period in nanoseconds (default: 12.5)
            hidden_layers: Tuple of hidden layer sizes (default: (256, 128, 64, 32))
            learning_rate: Learning rate for Adam optimizer (default: 0.001)
            max_iter: Maximum training iterations (default: 500)
            random_state: Random seed for reproducibility (default: 42)
        """
        self.laser_period = laser_period
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.is_fitted = False
        
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            learning_rate_init=learning_rate,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            verbose=False
        )
    
    def _augment_with_phasor(self, X):
        """Add phasor coordinates to input features."""
        G, S = compute_phasor_batch(X, self.laser_period)
        return np.column_stack([X, G, S])
    
    def fit(self, X, y, verbose=True):
        """
        Train the model on decay curve data.
        
        Args:
            X: Array of shape (n_samples, n_bins) containing decay histograms
            y: Array of shape (n_samples, 2) containing [tau1, tau2] labels
            verbose: Whether to print progress (default: True)
        
        Returns:
            self: The fitted model
        """
        if verbose:
            print(f"Augmenting {X.shape[0]} curves with Phasor coordinates...")
        
        X_augmented = self._augment_with_phasor(X)
        
        if verbose:
            print(f"Training Phasor-Fusion MLP on {X_augmented.shape[1]} features...")
        
        self.model.set_params(verbose=verbose)
        self.model.fit(X_augmented, y)
        self.is_fitted = True
        
        if verbose:
            print("Training complete.")
        
        return self
    
    def predict(self, X):
        """
        Predict fluorescence lifetimes from decay curves.
        
        Args:
            X: Array of shape (n_samples, n_bins) containing decay histograms
        
        Returns:
            Array of shape (n_samples, 2) containing [tau1, tau2] predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first or load a pretrained model.")
        
        X_augmented = self._augment_with_phasor(X)
        return self.model.predict(X_augmented)
    
    def score(self, X, y):
        """
        Compute R² score on test data.
        
        Args:
            X: Array of shape (n_samples, n_bins) containing decay histograms
            y: Array of shape (n_samples, 2) containing ground truth lifetimes
        
        Returns:
            R² score (1.0 is perfect)
        """
        X_augmented = self._augment_with_phasor(X)
        return self.model.score(X_augmented, y)


def analyze(decay_curve, model=None, laser_period=12.5):
    """
    Analyze a single decay curve to extract fluorescence lifetimes.
    
    This is a convenience function for quick analysis. For batch processing
    or custom models, use the TurboFLIM class directly.
    
    Args:
        decay_curve: 1D array of photon counts per time bin
        model: Optional pre-trained TurboFLIM model. If None, uses phasor analysis.
        laser_period: Laser repetition period in nanoseconds (default: 12.5)
    
    Returns:
        tuple: (tau1, tau2) estimated lifetimes in nanoseconds
    
    Example:
        >>> tau1, tau2 = analyze(my_decay_curve)
        >>> print(f"Short lifetime: {tau1:.2f} ns, Long lifetime: {tau2:.2f} ns")
    """
    from .phasor import compute_phasor, phasor_to_lifetime
    
    decay_curve = np.asarray(decay_curve)
    
    if model is not None:
        # Use trained model
        prediction = model.predict(decay_curve.reshape(1, -1))
        return float(prediction[0, 0]), float(prediction[0, 1])
    else:
        # Fallback to simple phasor analysis
        G, S = compute_phasor(decay_curve, laser_period)
        tau = phasor_to_lifetime(G, S, laser_period)
        return tau, tau  # Single-exponential approximation


def analyze_batch(image_stack, model=None, laser_period=12.5, verbose=True):
    """
    Analyze a stack of decay curves (e.g., from a FLIM image).
    
    Args:
        image_stack: Array of shape (n_pixels, n_bins) or (height, width, n_bins)
        model: Optional pre-trained TurboFLIM model
        laser_period: Laser repetition period in nanoseconds (default: 12.5)
        verbose: Whether to print progress (default: True)
    
    Returns:
        Array of shape (n_pixels, 2) or (height, width, 2) containing lifetimes
    
    Example:
        >>> lifetimes = analyze_batch(flim_image)
        >>> tau1_map = lifetimes[:, :, 0]
        >>> tau2_map = lifetimes[:, :, 1]
    """
    original_shape = image_stack.shape
    
    # Flatten if 3D image
    if len(original_shape) == 3:
        height, width, n_bins = original_shape
        image_stack = image_stack.reshape(-1, n_bins)
    
    n_pixels = image_stack.shape[0]
    
    if verbose:
        print(f"Analyzing {n_pixels} pixels...")
    
    if model is not None:
        lifetimes = model.predict(image_stack)
    else:
        # Fallback to phasor analysis
        from .phasor import compute_phasor_batch, phasor_to_lifetime
        G, S = compute_phasor_batch(image_stack, laser_period)
        tau = np.array([phasor_to_lifetime(g, s, laser_period) for g, s in zip(G, S)])
        lifetimes = np.column_stack([tau, tau])
    
    # Reshape back if was 3D
    if len(original_shape) == 3:
        lifetimes = lifetimes.reshape(height, width, 2)
    
    if verbose:
        print(f"Analysis complete. Mean lifetime: {np.mean(lifetimes):.2f} ns")
    
    return lifetimes
