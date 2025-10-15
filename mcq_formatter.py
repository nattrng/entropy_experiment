"""inputs three strings --> then formats it to pass to an llm for eval | simple code"""

def render_mc(question, letters, choices):
    prompt = question + "\n"
    for letter, choice in zip(letters, choices):
        prompt += f"{letter}. {choice}\n"
    return prompt.strip()