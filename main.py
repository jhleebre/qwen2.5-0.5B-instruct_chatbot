from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# ANSI color codes for terminal output formatting
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_status(message):
    """Print a formatted status message with blue color and bold text."""
    print(f"{Colors.OKBLUE}{Colors.BOLD}{'=' * 40}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}{Colors.BOLD}{message}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}{Colors.BOLD}{'=' * 40}{Colors.ENDC}\n")

def initialize_model(model_name):
    """Initialize the language model and tokenizer."""
    print("Initializing the model. This may take a moment...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Model initialization complete.")
    return model, tokenizer

def generate_response(model, tokenizer, messages):
    """Generate a response using the language model."""
    # Apply chat template to format the conversation history
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Create a TextStreamer for token-by-token output
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Generate the response
    _ = model.generate(
        **model_inputs,
        max_new_tokens=8192,
        streamer=streamer
    )

    return streamer.full_response, streamer.total_tokens

def trim_conversation(tokenizer, messages, max_tokens=120000):
    """Trim the conversation history to fit within the token limit."""
    while True:
        total_tokens = sum(len(tokenizer.encode(msg["content"])) for msg in messages)
        if total_tokens <= max_tokens:
            break
        if len(messages) <= 2:  # Keep system message and at least one user message
            break
        messages.pop(1)  # Remove the oldest message (after the system message)
    return messages

def get_multiline_input():
    """Get multi-line input from the user."""
    print("Enter your message (press Enter twice to finish):")
    lines = []
    while True:
        line = input()
        if line == "":
            if lines:  # End input only if there's content
                break
            else:
                print("Please enter at least one line.")
        else:
            lines.append(line)
    return "\n".join(lines)

class TextStreamer:
    """Stream text output token by token."""
    def __init__(self, tokenizer, skip_prompt=False, skip_special_tokens=False):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.skip_special_tokens = skip_special_tokens
        self.tokens = []
        self.started_streaming = False
        self.full_response = ""
        self.total_tokens = 0

    def put(self, value):
        """Process and print new tokens."""
        if self.skip_prompt and not self.started_streaming:
            self.started_streaming = True
            return
        self.tokens.extend(value.tolist())
        text = self.tokenizer.decode(self.tokens, skip_special_tokens=self.skip_special_tokens)
        print(text, end="", flush=True)
        self.full_response += text
        self.total_tokens += len(self.tokens)  # Update total token count
        self.tokens = []

    def end(self):
        """Finalize the streaming process."""
        if self.tokens:
            text = self.tokenizer.decode(self.tokens, skip_special_tokens=self.skip_special_tokens)
            print(text, end="", flush=True)
            self.full_response += text
            self.total_tokens += len(self.tokens)  # Update total token count
        print()  # Add a newline

def main():
    """Main function to run the chatbot."""
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model, tokenizer = initialize_model(model_name)
    
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}
    ]
    
    print_status("Welcome! Type 'bye', 'exit' or 'quit' on a new line to end the conversation.")
    
    while True:
        user_input = get_multiline_input()
        if user_input.lower() in ['bye', 'exit', 'quit']:
            print_status("Goodbye!")
            break
        
        print_status("Processing your input...")
        messages.append({"role": "user", "content": user_input.rstrip('\n')})  # Remove trailing newline
        messages = trim_conversation(tokenizer, messages)
        
        print(f"{Colors.OKGREEN}Qwen: {Colors.ENDC}", end="", flush=True)
        
        start_time = time.time()
        
        response, total_tokens = generate_response(model, tokenizer, messages)
        
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        tokens_per_second = total_tokens / elapsed_time if elapsed_time > 0 else 0
        
        print()
        print_status(
            f"Elapsed time: {elapsed_time:.2f} seconds\n"
            f"Total tokens generated: {total_tokens} tokens\n"
            f"Tokens per second: {tokens_per_second:.2f} tokens/sec"
        )

        messages.append({"role": "assistant", "content": response})
        messages = trim_conversation(tokenizer, messages)

if __name__ == "__main__":
    main()
