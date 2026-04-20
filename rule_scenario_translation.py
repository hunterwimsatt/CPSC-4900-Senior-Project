import torch
from load_model import load_model
from postprocessing import clear_memory, clean_output

def generate_scenario(model, tokenizer, doc_text, compliant=True, max_new_tokens=512):
    """
    Prompts the LLM to generate a single compliant or non-compliant test scenario
    for a given rule using system and user chat templates.
    """

    # 1. Define System and User Prompts
    # The system prompt enforces the persona and strict output constraints.
    system_prompt = (
        "You are a precise system testing assistant. Your task is to analyze a rule "
        "and generate a single scenario for testing.\n"
        "CRITICAL INSTRUCTION: You must output ONLY the plain text of the scenario itself. "
        "Do not include any explanations, reasoning, introductory text, conversational filler, "
        "or markdown formatting. Do not even provide anything like \"scenario:\" Provide the scenario "
        "and nothing else. Do not provide a story or explanation. Only provide a list of facts. "
        "Please note that the scenario generated will be used to test LLM "
        "translations of the rule-based text into SMT code, so write your"
        "scenarios containing data that can be easily translated into SMT code."
    )

    # The user prompt provides the specific data and the immediate request.
    compliance_type = "compliant" if compliant else "non-compliant"
    user_prompt = f"Rule: \"{doc_text}\"\n\nGenerate exactly one {compliance_type} scenario."

    # Format as a standard chat history
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # 2. Tokenization via Chat Template
    # `add_generation_prompt=True` tells the tokenizer to append the assistant header
    # so the model knows it is its turn to speak. `return_dict=True` makes it compatible with **inputs.
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    # 3. Generation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 4. Decoding and Cleanup
    # Slice the output to ignore the tokens from the prompt
    input_length = inputs["input_ids"].shape[1]
    raw_output = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    # Strip any accidental whitespace or newlines the model might have added
    return raw_output.strip()


def generate_scenarios(model, tokenizer, doc_text, num_compliant=1, num_non_compliant=1, max_tokens=1024):
    """
    
    """

    # 1. Define System and User Prompts
    # The system prompt enforces the persona and strict output constraints.
    system_prompt = (
        "You are a precise system testing assistant. Your task is to analyze a rule "
        "and generate a single scenario for testing.\n"
        "CRITICAL INSTRUCTION: You must output ONLY the plain text of the scenario itself. "
        "Do not include any explanations, reasoning, introductory text, conversational filler, "
        "or markdown formatting. Do not even provide anything like \"scenario:\". "
        "Provide the scenario and nothing else. Do not provide a story "
        "or explanation. Only provide a list of facts. "
        "Please note that the scenario generated will be used to test LLM "
        "translations of the rule-based text into SMT code, so write your "
        "scenarios containing data that can be easily translated into SMT code. "
        "You may be given existing scenarios that you are asked to NOT replicate."
    )

    scenarios = {"compliant": [], "non-compliant": []}

    # The user prompt provides the specific data and the immediate request.
    for i in range(num_compliant + num_non_compliant):
        
        compliance_type = "compliant"
        if i >= num_compliant:
            compliance_type = "non-compliant"

        compliant_scenarios = scenarios["compliant"]
        non_compliant_scenarios = scenarios["non-compliant"]

        compliant_scenario_text = ""

        for compliant_scenario in compliant_scenarios:
            compliant_scenario_text += compliant_scenario + "\n\n"

        non_compliant_scenario_text = ""

        for non_compliant_scenario in non_compliant_scenarios:
            non_compliant_scenario_text += non_compliant_scenario + "\n\n"
            
        user_prompt = None
        
        if compliance_type == "compliant":
            user_prompt = f"Rule:\n\n{doc_text}\n\n"
            user_prompt += f"Generate exactly one compliant scenario."
            user_prompt += "" if compliant_scenario_text == "" else f"\n\nDo not replicate the following compliant scenarios:\n\n{compliant_scenario_text}"
        else:
            user_prompt = f"Rule:\n\n{doc_text}\n\n"
            user_prompt += f"Generate exactly one non-compliant scenario."
            user_prompt += "" if non_compliant_scenario_text == "" else f"\n\nDo not replicate the following non-compliant scenarios:\n\n{non_compliant_scenario_text}"
    
        # Format as a standard chat history
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
        # 2. Tokenization via Chat Template
        # `add_generation_prompt=True` tells the tokenizer to append the assistant header
        # so the model knows it is its turn to speak. `return_dict=True` makes it compatible with **inputs.
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)


        print("Generating scenario...")
    
        # 3. Generation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
    
        # 4. Decoding and Cleanup
        # Slice the output to ignore the tokens from the prompt
        input_length = inputs["input_ids"].shape[1]
        raw_output = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
        # Strip any accidental whitespace or newlines the model might have added
        if compliance_type == "compliant":
            scenarios["compliant"].append(raw_output.strip())
        else:
            scenarios["non-compliant"].append(raw_output.strip())

    print("Scenarios generated!")

    del model
    del tokenizer
    del inputs
    del outputs
    clear_memory()
    print(f"--- Unloaded scenario model ---")
    
    return scenarios
    

def multi_rule_scenario_translation(model_name, model_id, rules_with_scenarios, max_tokens=1024):
    print(f"--- Loading {model_name} ---")

    # Load Tokenizer & Model
    model, tokenizer = load_model(model_id)

    results = []

    for idx, rule_with_scenario in enumerate(rules_with_scenarios):
        
        print(f"  Processing rule and scenario on {model_name}...")

        system_prompt = """You are an expert in SMT-LIB v2 (Z3). Your task is to translate a set of rules and a specific scenario into an SMT-LIB script where the solver's output (`sat` or `unsat`) dictates compliance or validity. 

**CRITICAL SMT-LIB ARCHITECTURE:**
1. **No Logic Declaration:** Do NOT use `(set-logic ...)`. Allow Z3 to automatically infer the correct logic based on the operators used. Forcing a restricted logic will crash the solver.
2. **No Status Variables:** Do NOT declare a boolean variable for "Compliant", "Eligible", "Approved", "Valid", or similar outcome concepts. 
3. **Declare Constants First:** You MUST begin your script by declaring all necessary constants for the scenario, including their sort (e.g., `(declare-const age Int)` or `(declare-const isLate Bool)`). Place all declarations at the top of the script.
4. **Rule Assertions:** Translate the provided rules into direct `(assert ...)` statements. These constraints must define what it mathematically takes to comply with the text.
5. **Disjunctive Rule Grouping:** Legal rules often contain baseline requirements alongside exceptions or alternative conditions. Do NOT stack absolute, contradictory `(assert)` statements (e.g., asserting a general rule must always be true, and then separately asserting an exception that contradicts it). Instead, you MUST group valid compliance paths together using an `(or)` statement to represent alternative ways to satisfy the overarching rules (e.g., `(assert (or (Standard_Path) (Exception_Path)))`).
6. **Scenario Assertions (No Assignments):** At the end of the script, bind the scenario values using strict equality assertions (e.g., `(assert (= age 30))`). Do NOT use invalid procedural commands like `(assign ...)`, `:=`, or `set`. 
7. **Do Not Pre-Compute:** Do NOT evaluate the scenario yourself. Do NOT negate assertions (e.g., `(assert (not ...))`) just to force a `sat` outcome. If the scenario facts violate the rule constraints, the solver returning `unsat` is the strictly desired and correct behavior.
8. **Execution:** End strictly with `(check-sat)` followed by your end token. Do not use `(get-model)`.
9. **Strict Comment Prohibition (No Scratchpad):** You are expressly forbidden from using SMT-LIB comments (`;`) to "think," "plan," or "calculate" your logic. Narrative comments, internal monologues, and step-by-step mathematical reasoning written as comments are treated as fatal syntax errors in this environment. The ONLY comments permitted in your entire output are the exact structural headers: `; Variables`, `; Rules`, and `; Scenario`.

**OUTPUT FORMAT:**
Output ONLY valid SMT-LIB code in PLAINTEXT. Do not use Markdown code blocks (no backticks). Do not explain outside of the code. 
**Structured Comments:** You MUST use standard SMT-LIB comments (starting with `;`) to structure your logic into three distinct sections: `; Variables`, `; Rules`, and `; Scenario`.

**EXAMPLES:**

Rule Text: A package is approved for standard air freight if it weighs less than 20 kg and is not classified as hazardous. If the package is 20 kg or heavier, it may still be approved for air freight if it has a special oversized waiver AND is not hazardous. Hazardous materials are never approved for air freight under any circumstances.
Scenario: Weight: 25, Hazardous: No, Waiver: Yes.
Output:
; Variables
(declare-const weight Int)
(declare-const isHazardous Bool)
(declare-const hasOversizedWaiver Bool)

; Rules
(assert (or 
  (and (< weight 20) (not isHazardous)) 
  (and (>= weight 20) hasOversizedWaiver (not isHazardous))
))

; Scenario
(assert (= weight 25))
(assert (= isHazardous false))
(assert (= hasOversizedWaiver true))

(check-sat)

Rule Text: A customer can return an item for a full refund if the item was purchased within the last 30 days and is unopened. If the item is defective, it can be returned for a full refund up to 90 days after purchase, regardless of being opened.
Scenario: Days since purchase: 45, Opened: Yes, Defective: Yes.
Output:
; Variables
(declare-const daysSincePurchase Int)
(declare-const isOpened Bool)
(declare-const isDefective Bool)

; Rules
(assert (or (and (<= daysSincePurchase 30) (not isOpened)) (and (<= daysSincePurchase 90) isDefective)))

; Scenario
(assert (= daysSincePurchase 45))
(assert (= isOpened true))
(assert (= isDefective true))

(check-sat)

Rule Text: To operate a commercial vehicle, a driver must be at least 21 years old and hold a valid Class A license. If the driver is transporting hazardous materials, they must additionally possess an active Hazmat endorsement.
Scenario: Driver age: 25, Class A license: Yes, Transporting Hazmat: Yes, Hazmat endorsement: No.
Output:
; Variables
(declare-const driverAge Int)
(declare-const hasClassALicense Bool)
(declare-const transportsHazmat Bool)
(declare-const hasHazmatEndorsement Bool)

; Rules
(assert (and (>= driverAge 21) hasClassALicense))
(assert (=> transportsHazmat hasHazmatEndorsement))

; Scenario
(assert (= driverAge 25))
(assert (= hasClassALicense true))
(assert (= transportsHazmat true))
(assert (= hasHazmatEndorsement false))

(check-sat)"""

        # System / User Prompts
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": rule_with_scenario}
        ]

        # Apply Chat Template
        try:
            if "Gemma" in model_name or model_name == "Qwen3.5-35B-A3B":
                text_input = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    enable_thinking=False,
                    add_generation_prompt=True
                )

                if "Gemma" in model_name:
                    text_input += "\nNO THINKING ALLOWED"
            else:
                text_input = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
        except Exception as e:
            # Fallback for models without chat templates
            text_input = f"User: {messages[1]['content']}\nAssistant:"

        inputs = None
        if model_name == "Yi":
            inputs = tokenizer([text_input], return_tensors="pt").to(model.device)
        else:
            inputs = tokenizer(text_input, return_tensors="pt").to(model.device)

        print("Generating response...")

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<end_of_turn>"),
            tokenizer.convert_tokens_to_ids("<|end_of_turn|>"),
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        # Filter out any tokens that don't exist in this specific tokenizer version
        terminators = [t for t in terminators if t is not None]
        
        
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,        # Deterministic
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=terminators
        )


        print("Response generated!")

        # Decode only the new tokens
        input_len = inputs.input_ids.shape[1]
        output_ids = generated_ids[0][input_len:]
        response_text = tokenizer.decode(output_ids, skip_special_tokens=True)

        print(response_text)
        response_text = clean_output(response_text)

        results.append(response_text)

    # Delete model and tokenizer to free VRAM for the next loop
    del model
    del tokenizer
    del inputs
    del generated_ids
    clear_memory()
    print(f"--- Unloaded {model_name} ---")

    return results