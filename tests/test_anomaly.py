# tests/test_anomaly.py: Tests Bug Catcher for Sigma Security
# Ensures anomaly detection catches technical and spiritual bugs

import unittest
import logging
import numpy as np
from src.anomaly_detector import AnomalyDetector
from src.logger import setup_logger

class TestAnomalyDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        log_file = 'logs/sigma_security.log'
        setup_logger(log_file)
    
    def test_normal_data(self):
        detector = AnomalyDetector(contamination=0.03)
        data = [1, 2, 3, 4, 5]
        result = detector.detect(data)
        self.assertEqual(set(result), {1})  # All normal
    
    def test_anomalous_data(self):
        detector = AnomalyDetector(contamination=0.03)
        data = [1, 2, 1000, 4, 5]
        result = detector.detect(data)
        self.assertIn(-1, result)  # Detects anomaly

if __name__ == '__main__':
    unittest.main()
