import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from typing import Tuple, List
import logging
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from ..core.settings import N_COMPONENTS, RANDOM_STATE, EPS, N_JOBS

logger = logging.getLogger(__name__)

class ScalableVGM:
    """Scalable implementation of Variational Gaussian Mixture model"""
    
    def __init__(self, n_components: int = N_COMPONENTS, random_state: int = RANDOM_STATE):
        self.n_components = n_components
        self.random_state = random_state
        self.bgm = BayesianGaussianMixture(
            n_components=n_components,
            weight_concentration_prior_type='dirichlet_process',
            random_state=random_state
        )
        self.valid_component_indicator = None
        self.means = None
        self.stds = None
    
    def fit(self, data: np.ndarray) -> 'ScalableVGM':
        """Fit the VGM model on input data"""
        try:
            # Reshape data if needed
            data = data.reshape(-1, 1) if len(data.shape) == 1 else data
            
            # Fit BGM
            self.bgm.fit(data)
            
            # Identify valid components (weight > eps)
            self.valid_component_indicator = self.bgm.weights_ > EPS
            
            # Store means and stds for valid components
            self.means = self.bgm.means_[self.valid_component_indicator].reshape(-1)
            self.stds = np.sqrt(self.bgm.covariances_[self.valid_component_indicator]).reshape(-1)
            
            logger.info(f"Fitted VGM with {sum(self.valid_component_indicator)} valid components")
            return self
            
        except Exception as e:
            logger.error(f"Error fitting VGM: {str(e)}")
            raise
    
    def transform_batch(self, data: np.ndarray) -> np.ndarray:
        """Transform a batch of data using mode-specific normalization"""
        try:
            data = data.reshape(-1, 1) if len(data.shape) == 1 else data
            
            # Get component assignments
            responsibilities = self.bgm.predict_proba(data)
            
            # Keep only valid components
            responsibilities = responsibilities[:, self.valid_component_indicator]
            
            # Get most likely component for each sample
            mode_indicators = responsibilities.argmax(axis=1)
            
            # Apply mode-specific normalization
            normalized_data = np.zeros_like(data)
            for idx, (mean, std) in enumerate(zip(self.means, self.stds)):
                mask = (mode_indicators == idx)
                normalized_data[mask] = (data[mask] - mean) / std
                
            return normalized_data
            
        except Exception as e:
            logger.error(f"Error transforming batch: {str(e)}")
            raise
    
    def inverse_transform_batch(self, data: np.ndarray, mode_indicators: np.ndarray) -> np.ndarray:
        """Inverse transform normalized data back to original scale"""
        try:
            data = data.reshape(-1, 1) if len(data.shape) == 1 else data
            
            # Apply inverse mode-specific normalization
            original_scale_data = np.zeros_like(data)
            for idx, (mean, std) in enumerate(zip(self.means, self.stds)):
                mask = (mode_indicators == idx)
                original_scale_data[mask] = data[mask] * std + mean
                
            return original_scale_data
            
        except Exception as e:
            logger.error(f"Error inverse transforming batch: {str(e)}")
            raise 