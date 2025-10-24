# Copyright Pathway Technology, Inc.

import os
import torch
import bdh

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

print(f"Using device: {device}")

# Load the trained model
BDH_CONFIG = bdh.BDHConfig(
    vocab_size=256,
    n_layer=2,
    n_head=2,
    n_embd=128,
    mlp_internal_dim_multiplier=64,
    dropout=0.1,
)

model = bdh.BDH(BDH_CONFIG).to(device=device, dtype=dtype)

# Try to load saved weights if they exist
checkpoint_path = os.path.join(os.path.dirname(__file__), "model.pt")
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print("Loaded model from checkpoint!")
else:
    print("No checkpoint found. You need to train the model first and save it.")
    print("Add this to train.py after training:")
    print('    torch.save(model.state_dict(), "model.pt")')
    exit(1)

model.eval()

print("\nInteractive Shakespeare Generator")
print("=" * 50)
print("Type your prompt and press Enter to generate text.")
print("Type 'quit' or 'exit' to stop.\n")

while True:
    try:
        user_input = input("Your prompt: ")

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not user_input:
            continue

        # Convert prompt to tensor
        prompt = torch.tensor(
            bytearray(user_input, "utf-8"),
            dtype=torch.long,
            device=device
        ).unsqueeze(0)

        # Generate text
        print("\nGenerating...", end=" ")
        with torch.no_grad():
            output = model.generate(
                prompt,
                max_new_tokens=200,  # Generate 200 characters
                top_k=5,             # More diverse output
                temperature=0.8      # Control randomness (lower = more focused)
            )

        # Decode and print
        result = bytes(output.to(torch.uint8).to("cpu").squeeze(0)).decode(
            errors="backslashreplace"
        )
        print("\n" + "=" * 50)
        print(result)
        print("=" * 50 + "\n")

    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        break
    except Exception as e:
        print(f"\nError: {e}\n")
