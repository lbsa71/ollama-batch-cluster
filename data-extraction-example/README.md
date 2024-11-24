# Clinical Notes Batch Data Extraction Example

***Note: the data in this directory is synthetic and contains no PII or PHI***  

This directory contains the configuration ([clinical-notes-config.toml](https://github.com/robert-mcdermott/ollama-batch-cluster/blob/main/data-extraction-example/clinical-notes-config.toml)), source data ([clinical-notes.jsonl](https://github.com/robert-mcdermott/ollama-batch-cluster/blob/main/data-extraction-example/clinical-notes.jsonl)) and example outputs ([clinical_nodes_data](https://github.com/robert-mcdermott/ollama-batch-cluster/tree/main/data-extraction-example/clinical_notes_data)) of a batch data extraction example. Using the provided configuration, it takes a collection of unstructured clinical notes that have been prepared and extracts data from each note using an LLM and puts it in JSON formatted output files, or singled combined JSON file ([clinical_notes_data.json](https://github.com/robert-mcdermott/ollama-batch-cluster/blob/main/data-extraction-example/clinical_notes_data.json)) using the provided [response-json-merge.py](https://github.com/robert-mcdermott/ollama-batch-cluster/blob/main/response-json-merge.py)script.

