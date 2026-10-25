import sys
from llama_cpp import Llama

# Set the path to your downloaded 4-bit GGUF file
MODEL_PATH = "llm_finetuning/base_model/sarvam-1.Q4_K_M.gguf"

def main():
    print("Loading Sarvam-1 model...")
    
    try:
        # Initialize the model
        llm = Llama(
            model_path=MODEL_PATH,
            verbose=False
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    print("Model loaded successfully.\n")

    # Sarvam-1 is a base model; best tested with prompt completion
    prompt = "कर्नाटक की राजधानी है:"

    print(f"Prompt: {prompt}")
    print("Generating response...\n")

    # Perform text completion
    output = llm(
        prompt=prompt,
        max_tokens=100,
        temperature=0.7,
        top_p=0.9
    )

    completed_text = output["choices"][0]["text"]
    print("--- Output ---")
    print(completed_text)

if __name__ == "__main__":
    main()