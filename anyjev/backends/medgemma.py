"""MedGemma image-and-text backend for ordinal clinical scoring."""

from pathlib import Path

import numpy as np


class MedGemmaBackend:
    def __init__(self, model_path, dtype="bfloat16"):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.torch = torch
        self.name = str(model_path)
        self.processor = AutoProcessor.from_pretrained(
            self.name, local_files_only=True
        )
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.name,
            dtype=getattr(torch, dtype),
            device_map="auto",
            local_files_only=True,
        )
        self.model.eval()

    def resolve_score_tokens(self, labels):
        """Validate score labels and return their single-token IDs."""
        ids = [
            self.tokenizer.encode(label, add_special_tokens=False)
            for label in labels
        ]

        if not ids or any(len(x) != 1 for x in ids):
            raise ValueError("Each score label must be a single token")

        token_ids = [x[0] for x in ids]

        if len(set(token_ids)) != len(token_ids):
            raise ValueError("Score token IDs must be distinct")

        return token_ids

    def score_image(self, image_path, prompt, labels=("0", "1", "2", "3", "4", "5")):
        from PIL import Image

        token_ids = self.resolve_score_tokens(labels)

        with Image.open(Path(image_path)) as im:
            image = im.convert("RGB")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        with self.torch.inference_mode():
            logits = self.model(**inputs).logits[0, -1, token_ids].float()
            probabilities = self.torch.softmax(logits, dim=-1)

        return probabilities.cpu().numpy().astype(np.float64)
