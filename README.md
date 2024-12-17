# Qwen2.5-0.5B-Instruct Chatbot

This program is an interactive AI chatbot interface using the Qwen2.5-0.5B-Instruct model. It's designed as a multi-turn conversation program running in a CLI environment, allowing easy testing of AI model conversations and simple performance evaluation.

## Key Features
- Initialization and loading of Qwen2.5-0.5B-Instruct model and tokenizer
- Multi-line input from users
- Real-time token-by-token response generation and output
- Conversation history management (max 120,000 tokens)
- Response generation performance monitoring (time taken, total tokens, tokens per second)
- Visual output distinction using ANSI color codes

## Installation
1. Install Python 3.12:
```sh
brew install python@3.12
```

2. Set up and activate virtual environment:
```sh
python3.12 -m venv venv_python312
source venv_python312/bin/activate
```

3. Install required libraries:
```sh
pip3 install torch torchvision
pip3 install transformers accelerate
pip3 install -U "huggingface_hub[cli]"
```

4. Download the model:
```sh
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir . --local-dir-use-symlinks False
```

## Usage
1. Run the program:
```sh
python test.py
```

2. Enter your message at the prompt (multi-line input supported)

3. Type 'bye', 'exit', or 'quit' to end the program

## Note
- This program has been tested on macOS Sequoia 15.1.1, Apple M3, with 8GB memory.
- Sufficient system resources may be required, and performance may vary depending on hardware.