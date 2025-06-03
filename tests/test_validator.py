# tests/test_validator.py: Tests Spiritual Gatekeeper for Sigma Security
# Ensures input validation catches bugs and enforces sigma discipline

import unittest
import logging
import numpy as np
from src.input_validator import InputValidator
from src.logger import setup_logger

class TestInputValidator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        log_file = 'logs/sigma_security.log'
        setup_logger(log_file)
    
    def test_valid_number(self):
        valid_input = 100
        result = InputValidator.validate(valid_input)
        self.assertEqual(result, valid_input)
    
    def test_invalid_type(self):
        with self.assertRaises(ValueError):
            InputValidator.validate("invalid")
    
    def test_chaotic_list(self):
        with self.assertRaises(ValueError):
            InputValidator.validate([1, 1000, -1000], strict=True)
    
    def test_missing_sigma_key(self):
        with self.assertRaises(ValueError):
            InputValidator.validate({"value": 10}, strict=True)

if __name__ == '__main__':
    unittest.main()
