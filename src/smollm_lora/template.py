"""Turn a list of chat messages into one training string."""

from typing import List


def apply_chat_template(messages: List) -> str:
    """Format user and assistant turns with a shared set of boundary tokens.

    An input such as ``[{"role": "user", "content": "hi there!"}, ...]``
    becomes::

        <|im_start|><|user|>: hi there!<|im_end|>
        <|im_start|><|assistant|>: hi, how is it going?<|im_end|>
    """
    lines = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "user":
            lines.append(f"<|im_start|><|user|>: {content}<|im_end|>\n")
        elif role == "assistant":
            lines.append(f"<|im_start|><|assistant|>: {content}<|im_end|>\n")
    return "".join(lines)
