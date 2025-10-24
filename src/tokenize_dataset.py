import torch
import tiktoken

import argparse
import numpy as np

def main(args):
    tokenizer = tiktoken.get_encoding("gpt2")
    with open(args.data_path, 'r') as f:
        dataset_txt = f.read()
    token_ids_list = tokenizer.encode(dataset_txt, allowed_special={"<|endoftext|>"})
    dataset = np.array(token_ids_list)
    np.save(args.output_path, dataset)
    print(f"Saved tokenized data to {args.output_path}, dtype={dataset.dtype}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tokenize dataset script")
    parser.add_argument('--data-path', type=str, required=True, help='Path to the data file to tokenize')
    parser.add_argument('--output-path', type=str, required=True, help='Path to the output file to save the tokenized dataset')
    args = parser.parse_args()
    main(args)