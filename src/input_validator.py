# src/input_validator.py: Spiritual Gatekeeper for Sigma Security
# Enforces strict validation to block chaotic or prideful inputs, acting as the sigma key to control access
# Locks down bugs and teaches spirits discipline with sigma swagger

import logging
import numpy as np

class InputValidator:
    """Gatekeeper for Sigma Security, validating inputs to prevent technical bugs and spiritual chaos."""
    
    @staticmethod
    def validate(data, strict=True):
        """
        Validate inputs to ensure sigma alignment, rejecting chaotic or unauthorized data.
        
        Args:
            data: Input data (int, float, list, dict) to validate.
            strict (bool): If True, enforce stricter sigma-specific checks.
        
        Returns:
            Validated data if it passes, else raises ValueError.
        
        Raises:
            ValueError: If input is invalid (chaotic, prideful, or unauthorized).
        """
        try:
            # Define expected types and numerical range
            expected_types = (int, float, list, dict, np.ndarray)
            valid_range = (-500, 500)  # Prevent extreme values causing bugs
            
            # Check type
            if not isinstance(data, expected_types):
                raise ValueError(f"Invalid input type: {type(data)}. Expected {expected_types}")
            
            # Validate numerical inputs
            if isinstance(data, (int, float)):
                if not (valid_range[0] <= data <= valid_range[1]):
                    raise ValueError(f"Input out of sigma range: {data}. Must be in {valid_range}")
            
            # Validate lists/arrays
            elif isinstance(data, (list, np.ndarray)):
                data_array = np.array(data, dtype=float)
                if np.any(~np.isfinite(data_array)):
                    raise ValueError("Chaotic spirit detected: Contains NaN or infinite values")
                if np.any((data_array < valid_range[0]) | (data_array > valid_range[1])):
                    raise ValueError(f"List contains values outside sigma range: {data_array}")
                if strict and np.std(data_array) > 100:  # Flag chaotic variance
                    raise ValueError(f"Input too chaotic: Standard deviation {np.std(data_array)} exceeds sigma limit")
            
            # Validate dictionaries
            elif isinstance(data, dict):
                if strict and 'sigma_key' not in data:
                    raise ValueError("Prideful spirit detected: Missing sigma_key for access")
                for key, value in data.items():
                    if not isinstance(value, (int, float, list, np.ndarray)):
                        raise ValueError(f"Invalid value for key {key}: {type(value)}")
                    InputValidator.validate(value, strict=strict)  # Recursive validation
            
            logging.info("Gatekeeper: Input validated with sigma discipline")
            return data
        
        except ValueError as e:
            logging.error(f"Gatekeeper: Denied chaotic or prideful input: {e}")
            raise
        except Exception as e:
            logging.error(f"Gatekeeper: Unexpected error validating input: {e}")
            raise ValueError(f"Validation failed: {e}")
