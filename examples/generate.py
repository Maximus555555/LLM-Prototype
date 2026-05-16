"""Example local generation script.

This uses an untrained tiny model by default, so output is random. Point the CLI
at a local checkpoint to generate from saved weights.
"""

from llm_prototype.inference import build_untrained_model, generate_text
from llm_prototype.tokenizer import ByteTokenizer


model = build_untrained_model("configs/tiny.json")
tokenizer = ByteTokenizer()
print(generate_text(model, tokenizer, "Once upon a time", max_new_tokens=40))
