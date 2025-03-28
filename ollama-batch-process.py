import asyncio
import json
import toml
import argparse
import random
import time
import re
from datetime import datetime
from ollama import AsyncClient
from pathlib import Path
import trafilatura
from urllib.parse import urlparse
import aiohttp
from typing import List, Tuple, Dict

def log_message(message):
    """Prints a message with a timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp}, {message}", flush=True)

async def fetch_url_content(url: str) -> str:
    """Fetch and clean content from a URL using trafilatura."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded)
            if text:
                return text.strip()
    except Exception as e:
        log_message(f"Error fetching {url}: {e}")
    return ""

def extract_markdown_links(text: str) -> List[Tuple[str, str]]:
    """Extract markdown links from text."""
    pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    return re.findall(pattern, text)

def process_prompt_with_context(prompt: str) -> Tuple[str, List[str]]:
    """Process a prompt, extracting links and preparing context."""
    links = extract_markdown_links(prompt)
    if not links:
        return prompt, []

    # Replace links with reference numbers
    for i, (text, url) in enumerate(links, 1):
        prompt = prompt.replace(f'[{text}]({url})', f'{text}[{i}]')

    return prompt, [url for _, url in links]

async def fetch_all_contexts(urls: List[str]) -> List[str]:
    """Fetch content from all URLs concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url_content(url) for url in urls]
        return await asyncio.gather(*tasks)

def create_context_block(contents: List[str]) -> str:
    """Create a formatted context block from fetched contents."""
    if not contents:
        return ""
    
    # Get current date in unambiguous format
    current_date = datetime.now().strftime("%B %d, %Y")
    
    context_block = "# Context Information\n\n"
    context_block += f"📅 Current date: {current_date}\n\n"
    
    # Add a section for contextual knowledge
    context_block += "## 📚 Contextual Knowledge\n"
    context_block += "Below are the relevant context sections that will inform the article:\n\n"
    
    for i, content in enumerate(contents, 1):
        if content:
            context_block += f"### Context Section {i}\n"
            context_block += f"{content[:500]}...\n\n"
    
    # Add a section for writing instructions
    context_block += "## ✍️ Writing Instructions\n"
    context_block += "Please use the above context to inform your article. "
    context_block += "Remember to maintain a friendly, engaging tone while being thorough and accurate.\n\n"
    
    return context_block

def save_response(prompt, response_text, output_dir, prompt_id=None):
    """Saves the response to JSON and TXT files with a unique filename based on prompt ID."""
    # Use prompt_id as filename, or fallback to a timestamp if no ID provided
    if prompt_id:
        base_filename = prompt_id
    else:
        epoch = int(time.time())
        random_suffix = random.randint(1000, 9999)
        base_filename = f"{epoch}-{random_suffix}"

    # Extract think content if present
    think_match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
    think_content = think_match.group(1).strip() if think_match else ""
    
    # Remove think tags from response
    response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()

    # Create the output dictionary
    output_data = {
        "prompt": prompt,
        "response": response_text,
        "think": think_content
    }

    # Ensure the output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write the JSON file
    json_path = output_dir / f"{base_filename}.json"
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(output_data, json_file, ensure_ascii=False, indent=4)
        json_file.flush()

    # Write the TXT file with just the response
    txt_path = output_dir / f"{base_filename}.txt"
    with open(txt_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(response_text)
        txt_file.flush()

async def chat(system_msg, message, host, gpu_index, model, output_dir):
    try:
        # Start timing
        start_time = datetime.now()

        # Process prompt for markdown links and fetch context
        prompt_text = message['content']
        processed_prompt, urls = process_prompt_with_context(prompt_text)
        
        # Get prompt ID if present
        prompt_id = message.get('id')
        
        # Fetch context if there are URLs
        contexts = []
        expanded_prompt = prompt_text
        if urls:
            contexts = await fetch_all_contexts(urls)
            context_block = create_context_block(contexts)
            # Add context to system message
            system_msg['content'] = context_block + "\n" + system_msg['content']
            # Create expanded prompt with context
            expanded_prompt = context_block + "\n\n" + prompt_text

        # Update message with processed prompt
        message['content'] = processed_prompt

        # Make the API request using the specified host and model
        response = await AsyncClient(host=f"http://{host}").chat(
            model=model,
            messages=[system_msg, message]
        )

        # Stop timing
        end_time = datetime.now()

        # Extract the response content
        response_text = response['message']['content']

        # Save the response to JSON and TXT files with expanded prompt
        save_response(expanded_prompt, response_text, output_dir, prompt_id)

        # Calculate duration and word count
        duration = (end_time - start_time).total_seconds()
        word_count = len(response_text.split())

        # Calculate words per second (WPS)
        wps = word_count / duration if duration > 0 else 0

        # Log the GPU index, word count, duration, and words per second
        log_message(f"Host: {host}, GPU: {gpu_index}, Words: {word_count}, Duration: {duration:.2f}s, WPS: {wps:.2f}")

    except Exception as e:
        log_message(f"Error on host {host}: {e}")

async def worker(host, gpu_index, model, task_queue, output_dir):
    """Worker function to process tasks using the specified host."""
    while not task_queue.empty():
        system_msg, message = await task_queue.get()
        await chat(system_msg, message, host, gpu_index, model, output_dir)
        task_queue.task_done()

async def main(config_path, prompts_path, output_dir):
    # Get configuration  
    config = toml.load(config_path)
    model = config["model"]
    gpus = config["ollama_instances"]
    system_msg = json.loads(f'{{"role": "system", "content": {json.dumps(config.get("system_message"))}}}')

    # Load prompts from both JSONL file and directory if they exist
    prompts = []
    prompts_path = Path(prompts_path)
    
    # Process input file if it exists
    if prompts_path.is_file():
        log_message(f"Loading prompts from file: {prompts_path}")
        if prompts_path.suffix == '.json':
            with open(prompts_path, "r") as file:
                data = json.load(file)
                if isinstance(data, list):
                    prompts.extend(data)
                else:
                    prompts.append(data)
        else:  # JSONL file
            with open(prompts_path, "r") as file:
                for line in file:
                    if line.strip():
                        prompts.append(json.loads(line))
    
    # Process prompts directory if it exists
    prompts_dir = Path("prompts")
    if prompts_dir.is_dir():
        log_message(f"Loading prompts from directory: {prompts_dir}")
        for json_file in prompts_dir.glob("*.json"):
            with open(json_file, "r") as file:
                data = json.load(file)
                if isinstance(data, list):
                    prompts.extend(data)
                else:
                    prompts.append(data)
    
    if not prompts:
        raise ValueError("No prompts found in either JSONL file or prompts directory")

    log_message(f"Total prompts to process: {len(prompts)}")

    # Create an async queue and populate it with prompts
    task_queue = asyncio.Queue()
    for message in prompts:
        task_queue.put_nowait([system_msg, message])

    # Create a list of worker tasks, one for each Ollama instance 
    tasks = []
    for host, gpu_index in gpus.items():
        tasks.append(worker(host, gpu_index, model, task_queue, output_dir))

    # Await the completion of all worker tasks
    await asyncio.gather(*tasks)

async def load_prompts(prompts_file):
    """Load prompts from a file."""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, Loading prompts from file: {prompts_file}")
    
    if prompts_file.endswith('.json'):
        with open(prompts_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                return [data]
    else:  # JSONL file
        prompts = []
        with open(prompts_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    prompts.append(json.loads(line))
        return prompts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ollama Batch Processing Client")
    parser.add_argument("--config", type=str, default="config.toml", help="Path to the configuration TOML file")
    parser.add_argument("--prompts", type=str, required=True, help="Path to the JSONL file with prompts (optional if prompts directory exists)")
    parser.add_argument("--output_dir", type=str, default="responses", help="Directory to save the response JSON files")

    args = parser.parse_args()

    try:
        asyncio.run(main(args.config, args.prompts, args.output_dir))
        log_message("All prompts processed. Exiting...")
    except KeyboardInterrupt:
        log_message("Process interrupted by user. Exiting...")

