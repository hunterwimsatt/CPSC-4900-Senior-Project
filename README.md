# CPSC 4900 Senior Project

The following is an overview of each file and what it does.

### Pipeline.ipynb

The overarching Jupyter notebook file where the differential testing pipeline is ran.

### data_analysis.ipynb

The Jupyter notebook file where data analysis was ran after the differential testing pipeline ran.

### Dolphin(/Qwen)MicrosoftQwenJohnny(/Small/Medium)Normal and Sorted

These CSV files contain the differential testing pipeline run on the Johnny/Small/Medium examples. The normal contains them as they were ran in the pipeline, while sorted shows them sorted and matched by document-scenario pair for easy comparison.

### load_model.py

Contains the functions that loads in a model and tokenizer from Hugging Face using the transformers library.

### postprocessing.py

Contains two functions, one which clears disk and GPU RAM memory and the other which cleans an LLM's output to hopefully break it down just to the SMT-LIB script.

### rule_scenario_translation.py

File which contains two primary functions, the first which generates the scenarios, and the second which generates the SMT-LIB scripts from the scenarios and runs the Z3 solver. There are two other functions which are artifacts from the beginnings of this project, which run a single document-scenario pair on one model, or generate a single scenario.

### run_smt.py

Small function which runs the SMT script, outputs sat, unsat, or unknown, and also outputs our default generic "error" when necessary.

### translation_verification_pipeline.py

Old artifact which had a singular translation of a document to SMT code.

### verification_pipeline.py

Master function which calls all other functions in this project to run the differential testing pipeline.

