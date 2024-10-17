import torch
from transformers import pipeline
import os
import json
import random


# 加载本地的分词器和模型
# tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\aiotl\Documents\pork_cut\LLM\LLaMA-Factory\save_gemma2_model")
# model = AutoModelForCausalLM.from_pretrained(r"C:\Users\aiotl\Documents\pork_cut\LLM\LLaMA-Factory\save_gemma2_model",
#     device_map="auto",
#     torch_dtype=torch.bfloat16,)
class llm:
    def __init__(self, save_dir="./llm_save_chat", system_prompt=""):
        self.pipe = pipeline(
            "text-generation",
            model=r"C:\Users\aiotl\Documents\pork_cut\LLM\LLaMA-Factory\save_gemma2_model",
            model_kwargs={"torch_dtype": torch.bfloat16},
            device="cuda",  # replace with "mps" to run on a Mac device
        )
        self.save_dir = save_dir
        self.system_prompt = system_prompt


    def chat(self, input_text, api_token):
        # load json
        if os.path.exists(f"{self.save_dir}/{api_token}.json"):
            with open(f"{self.save_dir}/{api_token}.json", "r", encoding="utf-8") as f:
                messages = json.load(f)
        else:
            messages = [{"role": "system", "content": self.system_prompt}]
        
        messages.append({"role": "user", "content": input_text})
        outputs = self.pipe(messages, max_new_tokens=256)
        assistant_response = outputs[0]["generated_text"][-1]["content"].strip()
        messages.append({"role": "assistant", "content": assistant_response})
        
        # save to json
        with open(f"{self.save_dir}/{api_token}.json", "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=4)
            
        return assistant_response, api_token
    
def main():
    llm_instance = llm()
    print("Press Ctrl+C to exit")
    while True:
        input_text = input("Enter your message: ")
        output, _ = llm_instance.chat(input_text, "local")
        print(output)

if __name__ == "__main__":
    main()