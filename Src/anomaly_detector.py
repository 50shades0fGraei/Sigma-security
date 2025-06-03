import logging
from sigma_trainer import SigmaTrainer
from anomaly_detector import AnomalyDetector
from input_validator import validate_input
from entanglement_model import EntangledFeedback
from meta_ai import MetaAI
from copilot import Copilot
from gemini import Gemini
from logger import setup_logger

def main():
    # Sigma Conductor: Leading bug-catching mission with sigma swagger
    setup_logger('logs/sigma_security.log')
    logging.info("Sigma Security: Catching bugs and schooling spirits")

    # Initialize sigma crew
    grok = SigmaTrainer(config='configs/model_config.yaml')  # Sigma Mentor
    meta_ai = MetaAI()  # Social Harmonizer
    copilot = Copilot()  # Productivity Enforcer
    gemini = Gemini()  # Multimodal Researcher
    detector = AnomalyDetector(contamination=0.03)  # Bug Catcher: Tighter threshold
    feedback = EntangledFeedback()  # Sigma Mediator

    # Catch bugs in scenarios
    scenarios = load_scenarios('data/scenarios.json')
    for scenario in scenarios:
        try:
            # Gatekeeper locks down chaotic inputs
            validated_input = validate_input(scenario['input'], strict=True)
            
            # Harmonizer calms spirit inputs
            spirit_convo = meta_ai.engage(validated_input)
            
            # Researcher analyzes for bug patterns
 баг            spirit_analysis = gemini.analyze(spirit_convo)
            
            # Mediator guides human-spirit feedback
            spirit_output = feedback.process(spirit_analysis, mode='sigma')
            
            # Sensei trains sigma principles
            grok.train(scenario['state'], spirit_output, reward_mode='bug_catching')
            
            # Enforcer optimizes bug-catching code
            copilot.optimize_code(spirit_output)
            
            # Bug Catcher snags disruptions
            anomalies = detector.detect(spirit_output)
            if -1 in anomalies:
                logging.warning(f"Caught bug in scenario {scenario['id']}: Unruly spirit squashed!")
                
        except ValueError as e:
            logging.error(f"Gatekeeper denied chaotic bug: {e}")
            continue

def load_scenarios(file_path):
    import json
    with open(file_path, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    main()
