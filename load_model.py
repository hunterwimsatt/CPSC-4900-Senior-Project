import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Default BitsAndBytesConfig for efficiency - can be overwritten
bnb_config_default = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


def load_model(model_id, eval = True, logging = False, bnb_config = bnb_config_default):
    """
    Loads a Hugging Face causal language model and its tokenizer.

    By default, this function attempts to load the model using 4-bit quantization 
    to save memory. If CUDA is not available, it gracefully degrades to standard 
    CPU loading and ignores the BitsAndBytes configuration.

    Args:
        model_id (str): The Hugging Face repository ID of the model to load (e.g., 'mistralai/Mistral-7B-v0.1').
        eval (bool, optional): If True, sets the model to evaluation mode (`model.eval()`). Defaults to True.
        logging (bool, optional): If True, prints step-by-step loading progress to the console. Defaults to False.
        bnb_config (BitsAndBytesConfig, optional): The quantization configuration to use. Defaults to 4-bit NF4.

    Returns:
        tuple: A tuple containing the loaded (model, tokenizer). 
               If the loading process fails, returns (None, None).
    """
    if logging:
        print(f"Loading {model_id}...")

    # Load Tokenizer & Model
    try:
        if logging:
            print("Loading Tokenizer...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        # Set pad_token in case
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if logging:
            print(f"Loading {model_id}...")

        if not torch.cuda.is_available():
            bnb_config = None
            print("CUDA not available, using CPU. To utilize BitsAndBytesConfig, please use CUDA.")

        model = None
        
        if "AWQ" in model_id or "gpt-oss-20b" in model_id:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="cuda",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                attn_implementation="eager"
            )
        elif "gemma" in model_id:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="cuda",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="cuda",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )

        if eval:
            if logging:
                print("Setting model to eval mode...")
            model.eval()

        if logging:
            print(f"{model_id} loaded.")
    except Exception as e:
        print(f"Failed to load {model_id}: {e}")
        return None, None

    return model, tokenizer