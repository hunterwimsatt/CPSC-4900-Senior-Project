import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from postprocessing import clear_memory, clean_output

def translate_doc(model_name, model_id, doc_text, output_to_file=False, logging=False):
    """
    Translates a plaintext rule document into SMT code.

    This function, given a HuggingFace model_name and model_id alongside a plaintext rule document,
    translates the document into SMT code by using the base model given. It loads the tokenizer and
    then the model in 4bit to limit RAM usage and overly extreme precision. Models are prompted on
    a generic system and user prompt to only produce SMT code. The code is cleaned up for some thinking
    responses and syntax for code blocks, so that only the text returned is the generated code. Please
    note that some models may ignore instructions and generate explanations, or use tokens that are not
    parsed. Additionally, weak models may produce incorrect responses.

    Args:
        model_name (str): Given name of LLM model used for plaintext to SMT translation (e.g. "MyFavoriteQwenModel").
        model_id (str): HuggingFace ID of LLM model used for plaintext to SMT translation (e.g. "Qwen/Qwen2.5-Coder-7B-Instruct").
        doc_text (str): Plaintext rule document.
        output_to_file (str, optional): Outputs the text to a file (./Data/{model_name}_result.txt). Defaults to False.
        logging (str, optional): Logs checkpoints of document translation pipeline. Defaults to False.

    Returns:
        str: Translated document into SMT code.
    """

    if logging:
        print(f"Loading {model_name}...")

    # Load Tokenizer & Model
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16, # Compute in 16-bit, store in 4-bit
            bnb_4bit_quant_type="nf4",            # "Normal Float 4" is best for LLMs
            bnb_4bit_use_double_quant=True,       # Compresses the quantization constants
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        # Set pad_token in case
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Use eval version of model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).eval()
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return

    print(f"Translating document to SMT code...")
    
    # System / User Prompts
    messages = [
        {"role": "system", "content": "You are an expert in SMT-LIB v2 (Z3) code. Translate the user's rule document into valid SMT-LIB code. Output ONLY the code. Do not explain. Your code should use assertions. Note that this code will be added on to later when we have a compliant or non-compliant scenario to test. It MUST be SMT-LIB code."},
        {"role": "user", "content": f"Document text:\n\n {doc_text}"}
    ]

    # Apply Chat Template
    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text_input, return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,        # Deterministic
        repetition_penalty=1.1, # Avoid repetition
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # Decode only the new tokens
    input_len = inputs.input_ids.shape[1]
    output_ids = generated_ids[0][input_len:]
    response_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    # Clean up potential decoding artifacts (SentencePiece underscore)
    response_text = clean_output(response_text)

    # Output to file if requested
    if output_to_file:
        output_filename = f"./Data/{model_name}_result.txt"
        try:
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(response_text)
            if logging:
                print(f"Results successfully saved to {output_filename}")
        except IOError as e:
            print(f"Failed to write to file: {e}")

    # Delete model and tokenizer to free VRAM for the next loop
    del model
    del tokenizer
    clear_memory()
    if logging:
        print(f"Unloaded {model_name}.")

    return response_text


