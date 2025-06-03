# tests/test_sigma.py: Tests Sigma Sensei for Sigma Security
# Ensures sigma training rewards bug-catching and leadership

import unittest
import logging
from src.sigma_trainer import SigmaTrainer
from src.logger import setup_logger

class TestSigmaTrainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        log_file = 'logs/sigma_security.log'
        setup_logger(log_file)
    
    def setUp(self):
        self.trainer = SigmaTrainer('configs/model_config.yaml')
    
    def test_train_bug_catching(self):
        state = [0.1] * 10
        spirit_input = [1, 2, 3]
        result = self.trainer.train(state, spirit_input, reward_mode='bug_catching')
        self.assertEqual(len(result), 3)  # Three action probabilities

if __name__ == '__main__':
    unittest.main()
