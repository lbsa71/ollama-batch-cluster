# Ollama Batch Cluster

The code in this repository will allow you batch process a large number of LLM prompts across one or more Ollama servers concurrently and centrally collect all the responses. 

This project started after I tried to get Ollama to make full use of a system with four Nvidia L40S GPUs but failed. I adjusted the *OLLAMA_NUM_PARALLEL* and *OLLAMA_SCHED_SPREAD* environment variables, and while it was now using all four GPUs, it was only pushing each GPU to about 25% utilization, so it didn't run any faster than if I had only had one GPU. I then ran four independent Ollama servers, each one pinned to different GPU using the *CUDA_VISIBLE_DEVICES* variable, a created script to load balance prompt across the Ollama servers. After a bunch of testing and refinement I ended up with the system I've shared in this repo. For a larger test, I spun up Ollama servers on six more servers, each with 4 L40S GPUs for a total of 28 GPUs and 1.344TB of VRAM. It seems to work perfectly. I know that vLLM is probably a better option for performance than Ollama, but I really like Ollama and it make it very simple. I used this setup to keep all 28 GPUs over 90% utilized for over 24 hours to stress test new GPUs before they went into production (one GPU kept failing with double bit errors and needed to replaced).

The following sections will so you how to use the code in this repo to set this up and use it to run batch jobs against a number of GPUs/servers.

These instructions assume that you are already know how to install/use Ollama and are familiar with Python.

## Starting the Ollama servers

The first thing we'll need to do is start up the Ollama servers, one per GPU. If you only have one GPU, or one GPU per multiple servers, and Ollama is already running, you probably don't need to this. To start the Ollama servers, one per GPU, we are going to use the provided *ollama-batch-servers.sh* shell script. It only takes a single argument which is an integer indicating the number of GPUs in the system. Example on a server with 4 GPUs:

```bash
bash ollama-batch-servers.sh 4
```

***Insert image here***

## Preparing your prompts

You'll need to create a *JSONL* formatted file, with a single prompt per line. Example:

```JSON
{"role": "user", "content": "Analyze the reasons behind the collapse of the Western Roman Empire."}
{"role": "user", "content": "How did Roman architecture influence urban development in Europe?"}
{"role": "user", "content": "Compare and contrast the political systems of the Roman Republic and the Roman Empire."}
{"role": "user", "content": "Discuss the role of religion in the daily life of Roman citizens and its impact on the Empire."}
{"role": "user", "content": "Evaluate the effects of Roman conquest on the cultures of the conquered territories."}
{"role": "user", "content": "How did the Roman Empire maintain control over such a vast territory?"}
{"role": "user", "content": "Examine the relationship between the Roman Senate and the Emperor."}
{"role": "user", "content": "What technological innovations did the Romans contribute to modern society?"}
{"role": "user", "content": "Analyze the role of the Roman economy in sustaining the empire’s growth and stability."}
{"role": "user", "content": "Describe the causes and consequences of the Roman Empire's split into Eastern and Western regions."}
```

## Configuring the batch client

The configuration file is a TOML formatted file that includes the LLM model to use, the list of Ollama instances to run the prompts against, and the system message to provide the LLM that will determine how it responds to the prompts. Here is an example configuration file:

```TOML
model = "llama3.2"
system_message = """You are an alien that can only respond with strings of emoji characters to convey your answer."""

[ollama_instances]
#format: "hostname:port" = GPU index
"server1:11432" = 0
"server1:11433" = 1
"server2:11432" = 0
"server2:11433" = 1
"server3:11432" = 0
"server3:11433" = 1
"server4:11432" = 0
"server4:11433" = 1
```

# Running a batch job

Now that we have our servers running, prompts prepared and a configuration, it's time to process the prompts across the cluster of hosts and GPUs. To do that we'll use the provided *ollama-batch-process.py*. But first we'll need to install the required dependencies

```bash
pip install ollama toml
```

The following is the usage documentation for the client:

```
usage: ollama-batch-process.py [-h] [--config CONFIG] --prompts PROMPTS [--output_dir OUTPUT_DIR]

Ollama Batch Processing Client

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to the configuration TOML file
  --prompts PROMPTS     Path to the JSONL file with prompts
  --output_dir OUTPUT_DIR
                        Directory to save the response JSON files
```

Example running the client:

```
python ollama-batch-process.py --config config.toml --prompts prompts.jsonl --output_dir batch10
```


# retriving the responses

* JSON files
* response-printer.py