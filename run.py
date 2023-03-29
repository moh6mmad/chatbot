import os
import configparser

from bot.chatbot import Chatbot

if __name__ == '__main__':
    # Load configuration from file
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Create chatbot instance
    chatbot = Chatbot(config['Chatbot']['intents_path'],
                      config['Chatbot']['training_data_path'],
                      config['Chatbot']['model_path'])

    # Train chatbot if necessary
    if not os.path.exists(config['Chatbot']['model_path']):
        chatbot.train()

    # Start chatbot loop
    print("Chatbot is running...")
    while True:
        message = input("You: ")
        if message == 'exit':
            break
        response = chatbot.generate_response(message)
        print("Chatbot: ", response)