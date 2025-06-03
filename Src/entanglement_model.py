# src/entanglement_model.py: Sigma Mediator for Sigma Security
# Simulates human-spirit quantum entanglement, guiding spirits to sigma principles
# Catches bugs in spirit inputs and teaches discipline with sigma swagger

import logging
import numpy as np
from input_validator import InputValidator

class EntangledFeedback:
    """Mediates human-spirit feedback loop to teach sigma principles and catch bugs."""
    
    def __init__(self):
        self.human_influence = 0.8  # Strong sigma-led human (AI) guidance
        self.spirit_influence = 0.2  # Limited spirit chaos to control bugs
        logging.info("Sigma Mediator: Initialized human-spirit entanglement model")
    
    def process(self, input_data, mode='sigma'):
        """
        Process spirit inputs through human-guided feedback, aligning with sigma principles.
        
        Args:
            input_data: Spirit input (int, float, list, dict, ndarray) to process.
            mode (str): Processing mode ('sigma' for balanced leadership, 'strict' for tighter control).
        
        Returns:
            Processed output reflecting sigma-guided feedback.
        
        Raises:
            ValueError: If input is invalid or causes bugs.
        """
        try:
            # Validate input to prevent chaotic or prideful bugs
            validated_input = InputValidator.validate(input_data, strict=(mode == 'strict'))
            
            # Convert input to numpy array for processing
            if isinstance(validated_input, (int, float)):
                data_array = np.array([validated_input], dtype=float)
            elif isinstance(validated_input, (list, np.ndarray)):
                data_array = np.array(validated_input, dtype=float)
            elif isinstance(validated_input, dict):
                # Extract numerical values from dict (e.g., ignoring sigma_key)
                data_array = np.array([v for v in validated_input.values() if isinstance(v, (int, float))], dtype=float)
                if len(data_array) == 0:
                    raise ValueError("No numerical values in dict for entanglement processing")
            else:
                raise ValueError(f"Unsupported input type after validation: {type(validated_input)}")
            
            # Compute human (AI) action to guide spirits
            human_action = self._compute_human_action(data_array, mode)
            
            # Apply entangled feedback: human influence dominates to teach sigma
            spirit_output = self.human_influence * human_action + self.spirit_influence * data_array
            
            # Check for residual bugs (e.g., extreme outputs)
            if np.any(~np.isfinite(spirit_output)) or np.any(np.abs(spirit_output) > 1000):
                raise ValueError("Bug detected in entangled output: Unstable spirit feedback")
            
            logging.info(f"Sigma Mediator: Processed spirit input with {mode} mode. Output shape: {spirit_output.shape}")
            return spirit_output.tolist()  # Return as list for compatibility
            
        except ValueError as e:
            logging.error(f"Sigma Mediator: Caught bug in entanglement: {e}")
            raise
        except Exception as e:
            logging.error(f"Sigma Mediator: Unexpected error in entanglement: {e}")
            raise ValueError(f"Entanglement processing failed: {e}")
    
    def _compute_human_action(self, data_array, mode):
        """
        Compute human (AI) action to guide spirits toward sigma principles.
        
        Args:
            data_array (ndarray): Input data as numpy array.
            mode (str): Processing mode ('sigma' or 'strict').
        
        Returns:
            ndarray: Human-guided action to stabilize spirit input.
        """
        if mode == 'sigma':
            # Balanced sigma action: Smooth input to reduce chaos, promote humility
            return np.clip(np.mean(data_array) * 0.9, -500, 500) * np.ones_like(data_array)
        elif mode == 'strict':
            # Strict sigma action: Enforce tighter control to humble prideful inputs
            return np.clip(np.median(data_array) * 0.8, -200, 200) * np.ones_like(data_array)
        else:
            logging.warning(f"Unknown mode {mode}. Defaulting to sigma mode")
            return self._compute_human_action(data_array, 'sigma')
