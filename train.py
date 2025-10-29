# Copyright Pathway Technology, Inc.

import os
from contextlib import nullcontext

import bdh
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Force full precision for GTX 1650 Ti (no bfloat16/float16 support)
dtype = "float32"
ptdtype = torch.float32
ctx = nullcontext()  # No mixed precision

# Disable AMP/GradScaler because float32 doesn't need it
scaler = None

# Optional: allow TF32 (safe on Ampere+, won't affect GTX 1650 Ti)
# Using new API (PyTorch 2.9+)
try:
    torch.backends.cuda.matmul.fp32_precision = 'tf32'
    torch.backends.cudnn.conv.fp32_precision = 'tf32'
except AttributeError:
    pass  # Older PyTorch versions

print(f"Using device: {device} with dtype {dtype}")

# Configuration - Optimized for low-end hardware
BATCH_SIZE = 16           # Very small batch size to reduce memory usage
BLOCK_SIZE = 128         # Shorter sequences for less memory
LEARNING_RATE = 3e-4     # Standard learning rate
WEIGHT_DECAY = 0.1       # Weight decay for regularization
MAX_ITERS = 5000         # Number of training iterations
LOG_FREQ = 50            # How often to print loss

BDH_CONFIG = bdh.BDHConfig(
    vocab_size=256,                # byte-level
    n_layer=3,                     # Minimal layers for low-end hardware
    n_head=2,                      # Minimal attention heads
    n_embd=256,                    # Small hidden state
    mlp_internal_dim_multiplier=64,  # Reduced from default 128
    dropout=0.1,                   # Standard dropout
)


input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")

# Fetch the tiny Shakespeare dataset
def fetch_data():
    if not os.path.exists(input_file_path):
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(input_file_path, "w") as f:
            f.write(requests.get(data_url).text)

# Load a training batch
def get_batch(split):
    data = np.memmap(input_file_path, dtype=np.uint8, mode="r")
    if split == "train":
        data = data[: int(0.9 * len(data))]
    else:
        data = data[int(0.9 * len(data)) :]
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + BLOCK_SIZE]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [torch.from_numpy((data[i + 1 : i + 1 + BLOCK_SIZE]).astype(np.int64)) for i in ix]
    )
    x, y = x.to(device), y.to(device)
    return x, y

# Main training loop
if __name__ == "__main__":
    fetch_data()

    model = bdh.BDH(BDH_CONFIG).to(device=device, dtype=ptdtype)

    # Optional: remove compile() for older GPUs
    # model = torch.compile(model)  # Uncomment only if torch.compile works on your setup

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    loss_acc = 0
    loss_steps = 0

    for step in range(MAX_ITERS):
        x, y = get_batch("train")
        with ctx:
            logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_acc += loss.item()
        loss_steps += 1

        if step % LOG_FREQ == 0:
            print(f"Step: {step}/{MAX_ITERS} loss {loss_acc / loss_steps:.3f}")
            loss_acc = 0
            loss_steps = 0

    print("Training done! Saving model...")

    # Save the trained model in the same directory as this script
    model_path = os.path.join(os.path.dirname(__file__), "model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    print("\nGenerating a sample...")

    model.eval()
    prompt = torch.tensor(
        bytearray("To be or ", "utf-8"), dtype=torch.long, device=device
    ).unsqueeze(0)

    ret = model.generate(prompt, max_new_tokens=100, top_k=3)
    ret_decoded = bytes(ret.to(torch.uint8).to("cpu").squeeze(0)).decode(
        errors="backslashreplace"
    )
    print(ret_decoded)

    print("\n" + "=" * 60)
    print("To interact with the model, run: python interact.py")
    print("=" * 60)

