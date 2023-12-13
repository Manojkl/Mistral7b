import sys
import time
import random
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


"""========================================================================================================================"""
def get_distribution(logits, temperature):
    probs = torch.softmax(logits / (temperature + 1e-10), dim=-1)
    return probs


def sample(logits, temperature):
    probs = get_distribution(logits, temperature)
    return torch.multinomial(probs, num_samples=1)[0]


def sample_from_draft_model(model, initial_prompt_seq, new_tokens, temperature=1.0):
    fin_prompt_seq = initial_prompt_seq.detach().clone()
    out_logits = []

    for _ in range(new_tokens):
        sample_token_logits = model(fin_prompt_seq).logits[:, -1, :]
        sample_token = sample(sample_token_logits, temperature=temperature)
        fin_prompt_seq = torch.concat([fin_prompt_seq, sample_token[None, ...]], dim=-1)
        out_logits.append(sample_token_logits)

    out_logits = torch.stack(out_logits, dim=1)
    return fin_prompt_seq, out_logits
"""========================================================================================================================"""
def autoregressive_sampling(model, initial_prompt_seq, target_len, temperature=1.0):
    n = initial_prompt_seq.shape[-1]
    fin_prompt_seq = initial_prompt_seq.detach().clone()

    while n < target_len:
        sample_token_logits = model(fin_prompt_seq).logits[:, -1, :]
        sample_token = sample(sample_token_logits, temperature=temperature)
        fin_prompt_seq = torch.concat([fin_prompt_seq, sample_token[None, ...]], dim=-1)
        n += 1
    return fin_prompt_seq
"""========================================================================================================================"""
def speculative_sampling(
    target_model,
    draft_model,
    initial_prompt_seq,
    max_new_tokens,
    tokenizer,
    lookahead=4,
    temperature=1.0,
    debug=True,
):
    """
    Implementation of Algorithm 2 of the paper - Accelerating Large Language Model Decoding
    with Speculative Sampling (https://arxiv.org/abs/2302.01318)
    """
    assert initial_prompt_seq.shape[0] == 1, "Batch size should be 1"

    n = initial_prompt_seq.shape[-1]
    fin_prompt_seq = initial_prompt_seq.detach().clone()

    while n < max_new_tokens:
        n_orig = n
        N = fin_prompt_seq.shape[-1]
        draft_outputs, draft_logits = sample_from_draft_model(
            draft_model, fin_prompt_seq, new_tokens=lookahead, temperature=temperature
        )

        if debug:
            print(
                f"Possible continuations: {tokenizer.decode(draft_outputs[0,n_orig:], skip_special_tokens=True)}"
            )

        target_logits = target_model(draft_outputs).logits[:, -lookahead - 1 :, :]

        target_model_distribution = get_distribution(target_logits, temperature)
        draft_model_distribution = get_distribution(draft_logits, temperature)

        accepted_flag = 1

        for t in range(lookahead):
            numerator = target_model_distribution[:, t, draft_outputs[0, N + t]]
            denominator = draft_model_distribution[:, t, draft_outputs[0, N + t]]
            ratio = numerator / denominator
            uniform_distribution = torch.rand_like(numerator)
            ones_tensor = torch.ones_like(numerator)

            # Rejection Sampling
            ## Acceptance
            if (uniform_distribution < torch.min(ones_tensor, ratio)).any():
                fin_prompt_seq = torch.concat(
                    [fin_prompt_seq, draft_outputs[:, N + t].unsqueeze(dim=-1)], dim=-1
                )
                n += 1

            ## Rejection
            else:
                new_dist = (
                    target_model_distribution[:, t, :]
                    - draft_model_distribution[:, t, :]
                )
                new_dist = torch.max(torch.zeros_like(new_dist), new_dist)
                new_dist = new_dist / new_dist.sum(dim=-1, keepdim=True)
                token_id = torch.multinomial(new_dist, num_samples=1)[0]
                fin_prompt_seq = torch.concat(
                    [fin_prompt_seq, token_id[None, ...]], dim=-1
                )
                accepted_flag = 0
                break

        if accepted_flag == 1:
            sample_token = sample(target_logits[:, -1, :], temperature=temperature)
            fin_prompt_seq = torch.concat(
                [fin_prompt_seq, sample_token[None, ...]], dim=-1
            )

        if debug:
            print(
                f"Accepted continuations: {tokenizer.decode(fin_prompt_seq[0,n_orig:], skip_special_tokens=True)}"
            )

        n += 1

    return fin_prompt_seq
"""========================================================================================================================"""
parser = argparse.ArgumentParser(description="Speculative Sampling")
parser.add_argument(
    "--method",
    default="speculative",
    help="Sampling Method (autogressive / speculative)",
)
parser.add_argument("--prompt", required=True, help="Input prompt")
parser.add_argument(
    "--max_new_tokens", type=int, required=True, help="No. of max new tokens"
)
parser.add_argument(
    "--target_model",
    default="facebook/opt-13b",
    help="Target model (HF Causal LM model)",
)
parser.add_argument(
    "--draft_model", required=False, help="Draft model (HF Causal LM model)"
)
parser.add_argument("--temperature", default=0, type=float, help="Temperature")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

if args.method == "speculative":
    if args.draft_model is None:
        print("Draft model should be specified for Speculative Sampling")
        sys.exit(1)

    print("Using target model:", args.target_model)
    print("Using draft model:", args.draft_model)

    target_model = AutoModelForCausalLM.from_pretrained(args.target_model).to(device)
    draft_model = AutoModelForCausalLM.from_pretrained(args.draft_model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    start_time = time.time_ns()
    tokens = speculative_sampling(
        target_model,
        draft_model,
        initial_prompt_seq=inputs.input_ids,
        max_new_tokens=args.max_new_tokens,
        tokenizer=tokenizer,
        temperature=args.temperature,
        debug=False,
    )
    end_time = time.time_ns()

    new_tokens = len(tokens[0]) - len(inputs.input_ids)
    time_taken = (end_time - start_time) / 1_000_000_000

    print("time_taken speculative:", time_taken)
    print(tokenizer.decode(tokens[0]))
    print()
    print(f"Latency (Speculative Sampling): {new_tokens/time_taken:.2f} tok/s")

elif args.method == "autoregressive":
    print("Using target model:", args.target_model)

    target_model = AutoModelForCausalLM.from_pretrained(args.target_model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    start_time = time.time_ns()
    tokens = autoregressive_sampling(
        target_model,
        initial_prompt_seq=inputs.input_ids,
        target_len=args.max_new_tokens + len(inputs.input_ids),
        temperature=args.temperature,
    )
    end_time = time.time_ns()

    new_tokens = len(tokens[0]) - len(inputs.input_ids)
    time_taken = (end_time - start_time) / 1_000_000_000

    print("time_taken Autoregressive:", time_taken)
    print(tokenizer.decode(tokens[0]))
    print()
    print(f"Latency (Naive Autoregressive Sampling): {new_tokens/time_taken:.2f} tok/s")

else:
    print("Method should be either autoregressive / speculative")
