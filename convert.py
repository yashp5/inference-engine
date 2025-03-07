# Converts a model consisting of a huggingface config.json, tokenizer.json, and .safetensors weights into a .yalm file
# which:
# - Normalizes the config to a common format in the header
# - Combines any safetensors shards
# - Reads the token vocabulary into a simple format
# - (TODO) Performs quantization to fp8 if specified

import argparse
import os
import json
import safetensors
import torch

SUPPORTED_ARCHITECTURES = [
    "MistralForCausalLM",
]

class Metadata:
    def __init__(self, config):
        arch = config["architectures"][0]
        if arch not in SUPPORTED_ARCHITECTURES:
            raise Exception(f"Architecture {arch} not supported, must be one of: {SUPPORTED_ARCHITECTURES}")
        self.arch = arch
        if arch == "MistralForCausalLM":
            self.dim = config["hidden_size"]
            self.hidden_dim = config["intermediate_size"]
            self.head_dim = config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
            self.n_layers = config["num_hidden_layers"]
            self.n_heads = config["num_attention_heads"]
            self.n_kv_heads = config.get("num_key_value_heads", config["num_attention_heads"])
            self.vocab_size = config["vocab_size"]
            self.max_seq_len = config["max_position_embeddings"]
            self.bos_token_id = config["bos_token_id"]
            self.eos_token_id = config["eos_token_id"]
            self.rope_theta = config.get("rope_theta", 10000.0)
            self.rotary_dim = int(self.head_dim * config.get("partial_rotatory_factor", 1))
            self.norm_eps = config["rms_norm_eps"]
            self.norm_type = "rmsnorm"

            assert config["hidden_act"] in ["silu", "gelu"]
            self.act_type = config["hidden_act"]


    def to_dict(self):
        result = {}
        result["arch"] = self.arch
        if self.arch == "MistralForCausalLM":
            result["dim"] = str(self.dim)
            result["hidden_dim"] = str(self.hidden_dim)
            result["head_dim"] = str(self.head_dim)
            result["n_layers"] = str(self.n_layers)
            result["n_heads"] = str(self.n_heads)
            result["n_kv_heads"] = str(self.n_kv_heads)
            result["vocab_size"] = str(self.vocab_size)
            result["max_seq_len"] = str(self.max_seq_len)
            result["bos_token_id"] = str(self.bos_token_id)
            result["eos_token_id"] = str(self.eos_token_id)
            result["rope_theta"] = str(self.rope_theta)
            result["rotary_dim"] = str(self.rotary_dim)
            result["norm_eps"] = str(self.norm_eps)
            result["norm_type"] = str(self.norm_type)
            result["act_type"] = str(self.act_type)
        return result


def load_tokens(tokenizer_path, vocab_size):
    tokens = [""] * vocab_size
    scores = [0] * vocab_size
    with open(tokenizer_path, "r") as f:
        tokenizer = json.load(f)
    
    vocab = tokenizer["model"]["vocab"]
    assert len(vocab) <= vocab_size

    for t, i in vocab.items():
        tokens[i] = t

    for added in tokenizer["added_tokens"]:
        tokens[added["id"]] = added["content"]
    
    # Scores are negaitve merge indices so that earlier merges come first
    for i, m in enumerate(tokenizer["model"]["merges"]):
        t1, t2 = m.split()
        ti = vocab[t1 + t2]
        if scores[ti] == 0:
            scores[ti] = -(1+i)

    # Preprocess tokens into UTF-8 encoding
    for i, t in enumerate(tokens):
        t = t.replace('\u2581', ' ') # sentencepiece uses this character to indicate a space
        b = t.encode("utf-8")
        b = b.replace(b"\0", b"\7") # replace null bytes with bell characters
        assert b.count(0) == 0
        tokens[i] = b

    return tokens, scores


def load_weights(model_files, dtype_str, metadata, tie_word_embeddings):
    weights = {}
    for model_path in model_files:
        ext = os.path.splitext(model_path)[1]
        if ext == ".safetensors":
            with safetensors.safe_open(model_path, framework="pt") as f:
                for k in f.keys():
                    assert(k not in weights)
                    weights[k] = f.get_tensor(k)


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("output", type=str)
    argp.add_argument("input", type="str", nargs="?")
    argp.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp8"])
    args = argp.parse_args()

    if args.input is None:
        # Input is a directory with HuggingFace layout, e.g. files:
        #   config.json
        #   tokenizer.json
        #   *.safetensors
        args.config = os.path.join(args.input, "config.json")
        if not os.path.exists(args.config):
            argp.error(f"config.json not found in {args.input}")
        
        args.tokenizer = os.path.join(args.input, "tokenizer.json")
        if not os.path.exists(args.tokenizer):
            argp.error(f"tokenizer.json not found in {args.input}")

        files = os.listdir(args.input)
        args.models = [os.path.join(args.input, f) for fname in files if os.path.splitext(fname)[1] == ".safetensors"]
        if len(args.models) == 0:
            argp.error(f"No safetensors files found in {args.input}")
    else:
        argp.error("Input must be a directory")

    with open(args.config, "r") as f:
        config = json.load(f)
        metadata = Metadata(config)

    tokens, scores = load_tokens(args.tokenizer, metadata.vocab_size)
    tensors = load_weights(args.models, args.dtype, metadata, config.get("tie_word_embeddings", None))

    # add tokenizer tensors at the end (to maximize the chance of model tensor alignment)
    # note: we concatenate all bytes of all tokens into a single tensor
    tensors["tokenizer_tokens"] = torch.cat([torch.tensor([x for x in b] + [0], dtype=torch.uint8) for b in tokens])
    tensors["tokenizer_scores"] = torch.tensor(scores, dtype=torch.float32)

    print(f"Saving {len(tensors)} tensors to {args.output}")
    safetensors.torch.save_file(tensors, args.output, metadata=metadata.to_dict())

            


