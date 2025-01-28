import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import sys

def main():
    print("Loading model and tokenizer...")
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" 
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while avoiding making up facts just to make the user happy. If you don't know the answer to something, say so."
    history = [f"System: {system_prompt}"]
    
    print(f"\nLoaded {model_name}")
    print("Type 'quit' to exit")
    print("-" * 50)

    while True:
        try:
            user_input = input("\nPrompt> ")
            
            if user_input.lower() == "quit":
                break
            if user_input.lower() == "clear":
                history = [f"System: {system_prompt}"]
                continue

            history.append(f"user: {user_input}")
            prompt = "\n".join(history) + "\nAssistant: "
            inputs = tokenizer(user_input, return_tensors="pt", return_attention_mask=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            print("\n", end='', flush=True)

            streamed_output = []
            for output in model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=2048,
                num_return_sequences=1,
                temperature=0.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                streamer=TextStreamer(tokenizer, skip_prompt=True),
            ):
                new_tokens = output[len(streamed_output):]
                streamed_output.extend(new_tokens)
                new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                print(new_text, end="", flush=True)

            print()
            assistant_response = tokenizer.decode(streamed_output, skip_special_tokens=True)
            history.append(f"Assistant: {assistant_response}")
            
        except KeyboardInterrupt:
            print("\nGeneration interrupted by user")
            break
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            break

if __name__ == "__main__":
    main()
