import torch
import gc
import os
import shutil

def clear_memory():
    """
    Clears existing memory from torch cuda objects and general unreachable objects.
    This helps avoid Out Of Memory issues.
    """
    
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.exists(cache_dir):
        for item in os.listdir(cache_dir):
            item_path = os.path.join(cache_dir, item)
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            except Exception as e:
                print(f"Failed to delete {item_path}: {e}")
                
    print("Cleared CUDA memory and Hugging Face disk cache.")


def clean_output(output):
    """
    Cleans the SMT code outputs of LLMs to only output the code itself.

    This function removes text from thinking/reasoning models, obfuscated tokens, and
    text tokens used for code blocks, so that the only text returned is the SMT code.
    Note that some models may have different methodologies for representing thinking,
    code blocks, etc., and may need additional processing after calling this function.

    Args:
        output (str): The LLM-generated text to be cleaned.

    Returns:
        str: Cleaned SMT code.
    """
    
    removable_strs = ["smt2", "smtlib", "`", "▁"]
    splittable_strs = ["</think>", "<channel|>"]
    for string in splittable_strs:
        if string in output:
            output = output.split(string)[-1]

    if "thought" in output:
        hold = output.split("thought")[-1].strip()

        if not hold and len(output.split("thought")) > 1:
            output = output.split("thought")[-2].strip() 
        else:
            output = hold           

    for string in removable_strs:
        output = output.replace(string, "")

    output = output.strip()

    return output