import re

def process_text(generated_text, remove_weights=True, remove_author=True):
    if remove_author:
        generated_text = re.sub(r'\bby:.*', '', generated_text)

    if remove_weights:
        generated_text = re.sub(r'\(([^)]*):[\d\.]*\)', r'\1', generated_text)
        generated_text = re.sub(r'(\w+):[\d\.]*(?=[ ,]|$)', r'\1', generated_text)

    # Remove markup tags
    generated_text = re.sub(r'<[^>]*>', '', generated_text)

    # Remove lonely symbols and formatting
    generated_text = re.sub(r'(?<=\s):(?=\s)', '', generated_text)
    generated_text = re.sub(r'(?<=\s);(?=\s)', '', generated_text)
    generated_text = re.sub(r'(?<=\s),(?=\s)', '', generated_text)
    generated_text = re.sub(r'(?<=\s)#(?=\s)', '', generated_text)

    # Clean up extra spaces and punctuation
    generated_text = re.sub(r'\s{2,}', ' ', generated_text)
    generated_text = re.sub(r'\.,', ',', generated_text)
    generated_text = re.sub(r',,', ',', generated_text)

    # Remove audio tags
    if '<audio' in generated_text:
        print(f"iF_prompt_MKR: Audio has been generated.")
        generated_text = re.sub(r'<audio.*?>.*?</audio>', '', generated_text)

    return generated_text.strip()