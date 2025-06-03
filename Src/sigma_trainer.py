import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import yaml
import logging

class SigmaTrainer:
    def __init__(self, config):
        with open(config, 'r') as f:
            self.config = yaml.safe_load(f)
        self.model = self._build_model()
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')
    
    def _build_model(self):
        # Sigma model: Built to catch bugs and lead
        return Sequential([
            Dense(64, activation='relu', input_shape=(10,), kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            Dense(3, activation='softmax')  # Actions: lead, support, squash_bug
        ])
    
    def train(self, state, spirit_input, reward_mode='bug_catching'):
        # Train AI and spirits to catch bugs with sigma swagger
        action_probs = self.model.predict(state.reshape(1, -1), verbose=0)
        reward = self._calculate_reward(action_probs, spirit_input, reward_mode)
        self.model.fit(state.reshape(1, -1), action_probs, sample_weight=[reward], verbose=0)
        logging.info(f"Sigma Sensei: Trained with {reward_mode} reward: {reward}")
        return action_probs
    
    def _calculate_reward(self, action_probs, spirit_input, reward_mode):
        # Reward sigma bug catching
        if reward_mode == 'bug_catching':
            if max(action_probs[0]) > 0.7:  # Penalize alpha-like overreach
                return -0.5
            if spirit_input.std() > 1.0:  # Penalize chaotic bugs
                return -0.3
            return 1.2  # Boost reward for catching bugs
        return 0.5
