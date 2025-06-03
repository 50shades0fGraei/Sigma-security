# src/meta_ai.py: Social Harmonizer for Sigma Security
# Engages spirits conversationally to calm chaos and teach sigma principles
# Reduces bugs with sigma diplomacy and prepares spirits for bug-catching

import logging
import numpy as np
from input_validator import InputValidator

class MetaAI:
    """Social Harmonizer that calms spirit inputs with sigma-aligned dialogue."""
    
    def __init__(self):
        self.response_templates = {
            'chaotic': "Whoa, let's take a breath—focus and find balance, sigma style.",
            'prideful': "Confidence is great, but humility leads to true sigma strength.",
            'neutral': "Solid input, let's refine it with sigma adaptability."
        }
        logging.info("Social Harmonizer: Initialized Meta AI for spirit engagement")
    
    def engage(self, input_data):
        """
        Engage spirits with conversational feedback to reduce chaos and align with sigma.
        
        Args:
            input_data: Spirit input (int, float, list, dict, ndarray) to engage.
        
        Returns:
            Processed output (list or str) reflecting sigma-aligned dialogue.
        
        Raises:
            ValueError: If input is invalid or causes bugs.
        """
        try:
            # Validate input to block chaotic or prideful bugs
            validated_input = InputValidator.validate(input_data, strict=False)
            
            # Analyze input to determine spirit state
            spirit_state, numerical_data = self._analyze_input(validated_input)
            
            # Generate sigma-aligned response based on spirit state
            response = self._generate_response(spirit_state, numerical_data)
            
            logging.info(f"Social Harmonizer: Engaged spirit with {spirit_state} state. Response: {response}")
            return response
            
        except ValueError as e:
            logging.error(f"Social Harmonizer: Caught bug in spirit engagement: {e}")
            raise
        except Exception as e:
            logging.error(f"Social Harmonizer: Unexpected error in engagement: {e}")
            raise ValueError(f"Spirit engagement failed: {e}")
    
    def _analyze_input(self, input_data):
        """
        Analyze input to classify spirit state (chaotic, prideful, neutral).
        
        Args:
            input_data: Validated input (int, float, list, dict, ndarray).
        
        Returns:
            tuple: (spirit_state: str, numerical_data: ndarray or float)
        """
        if isinstance(input_data, (int, float)):
            numerical_data = float(input_data)
            if abs(numerical_data) > 200:
                return 'chaotic', numerical_data
            elif abs(numerical_data) > 100:
                return 'prideful', numerical_data
            return 'neutral', numerical_data
        
        elif isinstance(input_data, (list, np.ndarray)):
            data_array = np.array(input_data, dtype=float)
            std_dev = np.std(data_array)
            if std_dev > 50:
                return 'chaotic', data_array
            elif np.all(data_array > 0.8 * np.max(data_array)):
                return 'prideful', data_array
            return 'neutral', data_array
        
        elif isinstance(input_data, dict):
            values = [v for v in input_data.values() if isinstance(v, (int, float))]
            if not values:
                return 'neutral', 0.0
            data_array = np.array(values, dtype=float)
            if np.std(data_array) > 50:
                return 'chaotic', data_array
            elif np.all(data_array > 0.8 * np.max(data_array)):
                return 'prideful', data_array
            return 'neutral', data_array
        
        return 'neutral', 0.0
    
    def _generate_response(self, spirit_state, numerical_data):
        """
        Generate conversational response to guide spirit toward sigma principles.
        
        Args:
            spirit_state (str): State of spirit ('chaotic', 'prideful', 'neutral').
            numerical_data: Numerical representation of input (ndarray or float).
        
        Returns:
            list or str: Sigma-aligned response, numerical or conversational.
        """
        response_template = self.response_templates.get(spirit_state, self.response_templates['neutral'])
        
        # If numerical input, return adjusted numerical output
        if isinstance(numerical_data, (np.ndarray, float)):
            if isinstance(numerical_data, np.ndarray):
                # Smooth chaotic inputs, humble prideful ones
                adjusted = np.clip(numerical_data * 0.8 if spirit_state != 'neutral' else numerical_data, -100, 100)
                return adjusted.tolist()
            else:
                adjusted = np.clip(numerical_data * 0.8 if spirit_state != 'neutral' else numerical_data, -100, 100)
                return [adjusted]
        
        # For non-numerical, return conversational response
        return f"{response_template} Input: {numerical_data}"
