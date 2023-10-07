import argparse
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM


def main(args):

    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token_id = 0

    model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        use_flash_attention_2=True
    )

    prompt = 'The president of India is'

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to('cuda')

    s = model.generate(input_ids, max_new_tokens=128)
    
    print(tokenizer.decode(s[0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str)

    args = parser.parse_args()

    main(args)