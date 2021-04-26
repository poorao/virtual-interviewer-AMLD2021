import random
from itertools import chain
import warnings

import torch
import torch.nn.functional as F

from transformers import GPT2LMHeadModel, GPT2Tokenizer


SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    "additional_special_tokens": ["<speaker1>", "<speaker2>"],
}


def format_input(qa, followup, tokenizer, lm_labels=False, with_eos=True):
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    sequence = [[bos]] + qa + [followup + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [
        [speaker2 if (len(sequence) - i) % 2 else speaker1] + s
        for i, s in enumerate(sequence[1:])
    ]

    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [speaker1] + [
        speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence[1:]) for _ in s
    ]
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["lm_labels"] = [-100] * len(instance["input_ids"])
    if lm_labels:
        instance["lm_labels"] = (
            ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]
        )
    return instance


def add_special_tokens_(model, tokenizer):
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(
        ATTR_TO_SPECIAL_TOKEN
    )  # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)


def top_filtering(
    logits, top_k=0.0, top_p=0.9, threshold=-float("Inf"), filter_value=-float("Inf")
):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert (
        logits.dim() == 1
    )  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1
        )

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(qa, tokenizer, model, args, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []

    for i in range(args["max_length"]):
        instance = format_input(qa, current_output, tokenizer, with_eos=False)

        input_ids = torch.tensor(
            instance["input_ids"], device=args["device"]
        ).unsqueeze(0)
        token_type_ids = torch.tensor(
            instance["token_type_ids"], device=args["device"]
        ).unsqueeze(0)

        logits = model(input_ids, token_type_ids=token_type_ids)
        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[0, -1, :] / args["temperature"]
        logits = top_filtering(logits, top_k=args["top_k"], top_p=args["top_p"])
        probs = F.softmax(logits, dim=-1)

        prev = (
            torch.topk(probs, 1)[1]
            if args["no_sample"]
            else torch.multinomial(probs, 1)
        )
        if i < args["min_length"] and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1:
                    warnings.warn(
                        "Warning: model generating special token with probability 1."
                    )
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output


def follow_up_generator(history, model, tokenizer, args):
    # Tokenize
    for i, raw_text in enumerate(history):
        history[i] = tokenizer.encode(str(raw_text))

    with torch.no_grad():
        out_ids = sample_sequence(history, tokenizer, model, args)

    out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
    return out_text
