# Ollama Batch Cluster

The code in this repository will allow you batch process a large number of LLM prompts across one or more Ollama servers concurrently and centrally collect all the responses. 

This project started after I tried to get Ollama to make full use of a system with four Nvidia L40S GPUs but failed. I adjusted the *OLLAMA_NUM_PARALLEL* and *OLLAMA_SCHED_SPREAD* environment variables, and while it was now using all four GPUs, it was only pushing each GPU to about 25% utilization, so it didn't run any faster than if I had only had one GPU. I then ran four independent Ollama servers, each one pinned to different GPU using the *CUDA_VISIBLE_DEVICES* variable, a created script to load balance prompt across the Ollama servers. After a bunch of testing and refinement I ended up with the system I've shared in this repo. For a larger test, I spun up Ollama servers on six more servers, each with 4 L40S GPUs for a total of 28 GPUs and 1.344TB of VRAM. It seems to work perfectly. I know that vLLM is probably a better option for performance than Ollama, but I really like Ollama and it make it very simple. I used this setup to keep all 28 GPUs over 90% utilized for over 24 hours to stress test new GPUs before they went into production (one GPU kept failing with double bit errors and needed to replaced).

The following sections will so you how to use the code in this repo to set this up and use it to run batch jobs against a number of GPUs/servers.

These instructions assume that you are already know how to install/use Ollama and are familiar with editing files, and running scripts. If you are new to Ollama and what to learn how to use it, see [my article](https://medium.com/p/913e50d6b7f0/) that covers the basics. 

## Starting the Ollama servers

The first thing we'll need to do is start up the Ollama servers, one per GPU. If you only have one GPU, or one GPU per multiple servers, and Ollama is already running, you probably don't need to this. To start the Ollama servers, one per GPU, we are going to use the provided [*ollama-batch-servers.sh*](https://github.com/robert-mcdermott/ollama-batch-cluster/blob/main/ollama-batch-servers.sh) shell script. It only takes a single argument which is an integer indicating the number of GPUs in the system.

**Usage:**

```bash
chmod +x ollama-batch-servers
./ollama-batch-servers.sh <number of gpus>
```

**Example on a system with 4 GPUs**:

 ![starting ollama servers](images/start-ollama-servers.png)

## Preparing your prompts

Next You'll need to create a *JSONL* formatted file, with a single prompt per line. The following is an example:

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

The configuration file is a TOML formatted file that includes the LLM model to use, the list of Ollama instances to run the prompts against, and the system message to provide the LLM that will determine how it responds to the prompts. Here is an example configuration file for a setup with 4 servers, each with 2 GPUs:

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

If you are just running this on your laptop with the single standard Ollama process running on the default port, your configuraiton file should look like this

```TOML
model = "llama3.2"
system_message = """You are an alien that can only respond with strings of emoji characters to convey your answer."""

[ollama_instances]
"127.0.0.1:11434" = 0
```

## Running a batch job

Now that we have our servers running, prompts prepared and a configuration, it's time to process the prompts across the cluster of hosts and GPUs. To do that we'll use the provided [*ollama-batch-process.py*](https://github.com/robert-mcdermott/ollama-batch-cluster/blob/main/response-printer.py). But first we'll need to install the required dependencies, the *ollama* and *toml* modules:

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

Example running submitting a batch or prompts for processing:

![ollama batch process example](images/ollama-batch-process-example.png)


# Collecting the responses 

The responses from each prompt will be written to the designated output directory (or the 'responses' directory if an output name wasn't provided) in JSON format, with a file per prompt. The name of the files are the Unix epoch time followed by a random element to avoid collision (example: *1732121830-4358.json*). The output data files include the prompt and response pair.

Example output file contents:

```JSON
{
    "prompt": "Why is the sky blue?",
    "response": "The sky appears blue because molecules in Earth's atmosphere scatter shorter blue wavelengths of sunlight more than longer wavelengths like red."
}
```

If you what to combine all the responses into a single output, the provided [*response-printer.py*](https://github.com/robert-mcdermott/ollama-batch-cluster/blob/main/response-printer.py) script will merge the responses into a single output. 

Usage:

```
python response-printer <directory of response JSON files>
```

The following is an example of the combined responses using emoji output to keep it short:

```
########################################
# File: 1732088793-9394.json
########################################
Prompt:
Discuss the influence of Roman culture on the development of Western civilization.

Response:
🏯💫🔥📚💡👑🤴🏻💪🌎️🕊️👸💃🏻🕺😍

########################################
# File: 1732088814-4841.json
########################################
Prompt:
How did the Roman Empire maintain control over such a vast territory?

Response:
🏛️🔒💪🚣‍♂️🌄📜👑💼

########################################
# File: 1732121810-9808.json
########################################
Prompt:
How did the Roman legal system shape the foundation of modern law?

Response:
🏛️📜🔒👮‍♂️💼🕊️🚫👫🤝
```
