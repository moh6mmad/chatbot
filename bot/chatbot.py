import os
import logging
import tensorflow as tf
from .model import Model

class ChatBot:
    def __init__(self):
        self.model = None
        self.logger = logging.getLogger('chatbot')
        self.logger.setLevel(logging.INFO)

        # Set up a console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Set up a file handler
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_file = os.path.join(log_dir, 'chatbot.log')
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # Set up a formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        # Add the handlers to the logger
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)

        self.load_model()

    def load_model(self):
        self.logger.info('Loading model...')
        self.model = Model()
        self.model.load()

    def respond(self, message):
        return self.model.predict(message)