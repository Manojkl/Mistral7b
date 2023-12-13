from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"


def autoregressive(
    prompt="Alice and Bob", model_id="mistralai/Mistral-7B-Instruct-v0.1"
):
    # print("autoregressive")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto").to(
        device
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, torch_dtype="auto")
    text = prompt
    inputs = tokenizer(text, return_tensors="pt").to(device)
    start = time.perf_counter()
    outputs = model.generate(**inputs, max_new_tokens=20)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    elapsed_time = time.perf_counter() - start
    print(elapsed_time)


def speculative_sampling(
    prompt="Alice and Bob",
    model_id="mistralai/Mistral-7B-Instruct-v0.1",
    draft_id="EleutherAI/pythia-160m-deduped",
):
    # print("speculative")
    prompt = prompt
    checkpoint = model_id
    assistant_checkpoint = draft_id
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    assistant_model = AutoModelForCausalLM.from_pretrained(assistant_checkpoint).to(
        device
    )
    start = time.perf_counter()
    outputs = model.generate(
        **inputs, assistant_model=assistant_model, max_new_tokens=20
    )
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    elapsed_time = time.perf_counter() - start
    print(elapsed_time)


if __name__ == "__main__":
    # if you type --help
    parser = argparse.ArgumentParser(description="Run some functions")

    # Add a command
    parser.add_argument("--prompt", help="Input prompt to LLM")
    parser.add_argument(
        "--autoregressive_model_id", help="Primary LLM model name from huggingface"
    )
    parser.add_argument(
        "--speculative_model_id", help="Primary LLM model name from huggingface"
    )
    parser.add_argument(
        "--speculative_draft_id",
        help="assistant lightweight LLM name from huggingface",
    )

    # Get our arguments from the user
    args = parser.parse_args()

    if args.autoregressive_model_id:
        autoregressive(args.prompt, args.autoregressive_model_id)

    if args.speculative_model_id:
        speculative_sampling(
            args.prompt, args.speculative_model_id, args.speculative_draft_id
        )
