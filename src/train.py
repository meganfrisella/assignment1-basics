import torch
import argparse
import utils
import transformer
import numpy as np
import numpy.typing as npt
import optimizer as optimizer
import os
import tiktoken
from torch.profiler import profile, ProfilerActivity
import time

NUMPY_HEADER_LEN = 128

def decode(model: transformer.TransformerLM, prompt: str, tokenizer: tiktoken.Encoding, max_length: int=50, temperature: float=1.0, top_p: float=1.0) -> str:
    in_tokens = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
    out_tokens = torch.tensor(in_tokens)
    for _ in range(max_length):
        logits = model(out_tokens)
        logits = utils.softmax(logits, dim=-1, temperature=temperature)
        probs = logits[-1]
        if top_p < 1.0:
            probs = utils.top_p_sampling(probs, top_p)
        next_token = torch.multinomial(probs, 1)
        out_tokens = torch.cat((out_tokens, next_token), dim=-1)
        if next_token == tokenizer.eot_token:
            break
    return tokenizer.decode(out_tokens.squeeze(-1).tolist())

def train_step(model: transformer.TransformerLM, optim: optimizer.AdamW, batch: torch.Tensor, target: torch.Tensor) -> float:
    optim.zero_grad()
    logits = model(batch)
    loss = utils.cross_entropy(logits, target)
    loss.backward()
    optim.step()
    return loss

def main(args):
    # Load the dataset
    dataset = np.memmap(args.data_path, dtype=np.int64, mode='r', offset=NUMPY_HEADER_LEN)

    print("Loaded dataset")

    # Load the model
    model = transformer.TransformerLM(args.vocab_size, args.context_length, args.d_model, args.num_layers, args.num_heads, args.d_ff, args.rope_theta)
    model.to(args.device)

    print("Loaded model")

    # Load the optimizer
    optim = optimizer.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2), eps=args.epsilon)

    # Load the checkpoint
    if args.load_checkpoint_path is not None:
        utils.load_checkpoint(args.load_checkpoint_path, model, optim)
        print(f"Loaded checkpoint from {args.load_checkpoint_path}")

    # Make the checkpoint directory
    if args.save_checkpoint:
        os.makedirs(args.save_checkpoint_dir, exist_ok=True)

    # Train the model
    batch, target = utils.get_batch(dataset, args.batch_size, args.context_length, args.device)
    print("Loaded batch")
    if args.profile:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
            for epoch in range(args.epochs):
                time_start = time.time()
                loss = train_step(model, optim, batch, target)
                time_end = time.time()
                if args.save_checkpoint and epoch % args.checkpoint_every == 0:
                    utils.save_checkpoint(model, optim, epoch, args.save_checkpoint_dir + f"/checkpoint_{epoch}.pt")
                if epoch % args.log_every == 0:
                    print(f"Epoch {epoch} time: {time_end - time_start:0.2f} s | loss: {loss.item():.2f} | perplexity: {utils.perplexity(loss).item():.2f}")
        prof.export_chrome_trace("trace.json")
    else:
        for epoch in range(args.epochs):
            time_start = time.time()
            loss = train_step(model, optim, batch, target)
            time_end = time.time()
            if args.save_checkpoint and epoch % args.checkpoint_every == 0:
                utils.save_checkpoint(model, optim, epoch, args.save_checkpoint_dir + f"/checkpoint_{epoch}.pt")
            if epoch % args.log_every == 0:
                print(f"Epoch {epoch} time: {time_end - time_start:0.2f} s | loss: {loss.item():.2f} | perplexity: {utils.perplexity(loss).item():.2f}")
    
    # Sample some prompts
    tokenizer = tiktoken.get_encoding("gpt2")
    out = decode(model, "Which kind of cat is best?", tokenizer, max_length=10, temperature=0.8, top_p=0.9)
    print(out)

def parse_args():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--data-path', type=str, required=True, help='Path to the training data')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--context-length', type=int, default=256, help='Context length for training')
    parser.add_argument('--vocab-size', type=int, default=50257, help='Vocabulary size for training')
    parser.add_argument('--d-model', type=int, default=512, help='Model dimension')
    parser.add_argument('--num-layers', type=int, default=4, help='Number of layers in the transformer')
    parser.add_argument('--num-heads', type=int, default=16, help='Number of heads in the transformer')
    parser.add_argument('--d-ff', type=int, default=1344, help='Feed-forward dimension')
    parser.add_argument('--rope-theta', type=float, default=10000.0, help='RoPE theta')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for AdamW optimizer')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 for AdamW optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 for AdamW optimizer')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay for AdamW optimizer')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='Epsilon for AdamW optimizer')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training (e.g., cpu, cuda:0)')
    parser.add_argument('--load-checkpoint-path', type=str, default=None, help='Path to load checkpoint from')
    parser.add_argument('--save-checkpoint', action='store_true', help='Checkpoint training')
    parser.add_argument('--save-checkpoint-dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--checkpoint-every', type=int, default=500, help='Checkpoint every N epochs')
    parser.add_argument('--log-every', type=int, default=10, help='Log every N epochs')
    parser.add_argument('--profile', action='store_true', help='Profile the training')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)

