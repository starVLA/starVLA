# Dependency-free helper shared by the in-forward tokenization path and the
# DataLoader preprocessing collate. Lives in its own leaf module so DataLoader
# workers can import it without pulling in the full model module.


def build_qwenvl_messages(images, instructions, cot_prompt=None, solutions=None):
    """Chat-template messages for a batch: one user message per sample."""
    messages = []
    assert len(images) == len(instructions), "Images and instructions must have the same length"
    for imgs, instruction in zip(images, instructions):
        content = [{"type": "image", "image": img} for img in imgs]
        prompt = cot_prompt.replace("{instruction}", instruction) if cot_prompt is not None else instruction
        content.append({"type": "text", "text": prompt})
        msg = [{"role": "user", "content": content}]
        if solutions is not None:
            msg.append({"role": "assistant", "content": [{"type": "text", "text": solutions[len(messages)]}]})
        messages.append(msg)
    return messages
