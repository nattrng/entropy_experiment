"""
HellaSwag dataset evaluator.
https://huggingface.co/datasets/hellaswag/hellaswag

Implements the same small interface as other evals in `evals/`:
- __init__(subset, split)
- num_examples()
- get_example(index) -> {"messages": [...], "letters": [...]}
- evaluate(conversation, assistant_response) -> bool

This dataset has 4 choices per example; we map them to letters A-D.
"""

from datasets import load_dataset
from mcq_formatter import render_mc


class HellaSwag:
    letters = ('A', 'B', 'C', 'D')

    def __init__(self, subset, split):
        # HellaSwag supports the 'default' config; keep checks similar to other evals
        assert split in ["train", "validation", "test", "dev"], "HellaSwag split must be train|validation|dev|test"
        # load and shuffle for reproducibility
        self.ds = load_dataset("hellaswag", split=split).shuffle(seed=42)

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        # HellaSwag rows contain: context, ending_candidates (list of 4), label (int 0-3)
        context = row.get("ctx") or row.get("context") or row.get("sent1", "")
        # candidates key may be 'endings' or 'ending_candidates'
        choices = row.get("ending_candidates") or row.get("endings") or row.get("choices")
        if choices is None:
            # fall back: try keys that sometimes appear
            choices = [row.get(k, "") for k in ("ending0", "ending1", "ending2", "ending3")]

        # ensure we have 4 choices
        assert len(choices) == 4, f"HellaSwag example must have 4 choices, found {len(choices)}"
        answer_idx = row.get("label")
        assert answer_idx is None or 0 <= answer_idx < 4, "HellaSwag label must be in 0..3 or None"

        # render MC-like prompt (context + choices)
        user_message = render_mc(context, self.letters, choices)
        assistant_message = self.letters[answer_idx] if answer_idx is not None else self.letters[0]

        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]

        return {"messages": messages, "letters": self.letters}

    def evaluate(self, conversation, assistant_response):
        assert assistant_response in self.letters, f"HellaSwag answer {assistant_response} is expected to be one of {self.letters}"
        assistant_message = conversation['messages'][-1]['content']
        return assistant_response == assistant_message
