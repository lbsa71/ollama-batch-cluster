import asyncio
import json
import toml
import argparse
import random
import time
import re
import os
from datetime import datetime
from ollama import AsyncClient
from pathlib import Path
import trafilatura
from urllib.parse import urlparse
import aiohttp
from typing import List, Tuple, Dict
import traceback

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

def save_response(prompt, response_text, output_dir, prompt_id=None, start_time=None, duration=None, length=None):
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
    
    # Clean up markdown output - remove article markers and think-related artifacts
    clean_markdown = response_text
    clean_markdown = re.sub(r'---BEGIN ARTICLE---\s*', '', clean_markdown, flags=re.DOTALL)
    clean_markdown = re.sub(r'---END ARTICLE---\s*', '', clean_markdown, flags=re.DOTALL)
    clean_markdown = re.sub(r'---END---\s*', '', clean_markdown, flags=re.DOTALL)
    clean_markdown = re.sub(r'\\boxed\{.*?\}', '', clean_markdown, flags=re.DOTALL)
    clean_markdown = re.sub(r'</think>', '', clean_markdown, flags=re.DOTALL)
    clean_markdown = re.sub(r'<CURRENT_CURSOR_POSITION>', '', clean_markdown, flags=re.DOTALL)
    clean_markdown = clean_markdown.strip()

    # Track the current timestamp for last_updated field
    current_time = datetime.now()

    # Create the output dictionary with timing and length metrics
    output = {
        "prompt": prompt,
        "response": response_text,
        "think": think_content,
        "metrics": {
            "start_time": start_time.isoformat() if start_time else None,
            "duration_seconds": duration,
            "length": length
        },
        "last_updated": current_time.isoformat()
    }

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save JSON with full data
    json_path = os.path.join(output_dir, f"{base_filename}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Save TXT with just the cleaned response
    txt_path = os.path.join(output_dir, f"{base_filename}.md")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(clean_markdown)

    return json_path, txt_path

def should_regenerate(prompt_id, output_dir, prompt_file_modified_time):
    """
    Check if we should regenerate the response by comparing prompt file's 
    modification time with the last_updated time in the output JSON.
    
    Returns True if the response should be regenerated, False otherwise.
    """
    if not prompt_id:
        return True  # Always regenerate if no prompt ID is provided
    
    # Check if the output file exists
    output_file = os.path.join(output_dir, f"{prompt_id}.json")
    if not os.path.exists(output_file):
        return True  # Regenerate if output doesn't exist
    
    try:
        # Load the existing output file
        with open(output_file, 'r', encoding='utf-8') as f:
            output_data = json.load(f)
        
        # Get the last_updated timestamp from the output
        last_updated_str = output_data.get('last_updated')
        if not last_updated_str:
            return True  # Regenerate if no last_updated field
        
        # Convert last_updated string to datetime
        last_updated = datetime.fromisoformat(last_updated_str)
        
        # Compare with prompt file's modification time
        if prompt_file_modified_time > last_updated:
            return True  # Regenerate if prompt file was modified after last run
        
        return False  # No need to regenerate
    except Exception as e:
        log_message(f"Error checking regeneration status: {str(e)}")
        return True  # Regenerate on error to be safe

async def chat(prompt, host, model, system_message, context_block=None):
    """Process a single prompt with context and return the response."""
    start_time = datetime.now()
    
    # Prepare messages with context if available
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    if context_block:
        messages.append({"role": "user", "content": context_block})
    messages.append({"role": "user", "content": prompt})

    # Make the API request
    client = AsyncClient(host=host)
    response = await client.chat(
        model=model,
        messages=messages,
        stream=False
    )
    
    # Calculate duration and get response length
    duration = (datetime.now() - start_time).total_seconds()
    length = len(response['message']['content'])
    
    return response['message']['content'], start_time, duration, length

async def worker(host, gpu_index, model, task_queue, output_dir):
    """Process prompts from the queue using the specified host and GPU."""
    while True:
        try:
            # Get a task from the queue
            task = await task_queue.get()
            if task is None:  # None is our signal to stop
                task_queue.task_done()
                break
                
            prompt_id, prompt, system_msg, prompt_file_modified_time = task
            
            # Check if we need to regenerate the response
            if not should_regenerate(prompt_id, output_dir, prompt_file_modified_time):
                log_message(f"Skipping {prompt_id} - prompt unchanged since last run")
                task_queue.task_done()
                continue
            
            start_time = datetime.now()
            try:
                # Process the prompt
                response_text, start_time, duration, length = await chat(prompt, host, model, system_msg)
                
                # Calculate words in response
                word_count = len(response_text.split())
                
                # Save response with timing information
                json_path, txt_path = save_response(
                    prompt, 
                    response_text, 
                    output_dir, 
                    prompt_id,
                    start_time,
                    duration,
                    length
                )
                
                # Log completion with metrics
                processing_time = (datetime.now() - start_time).total_seconds()
                wps = word_count / processing_time if processing_time > 0 else 0
                log_message(
                    f"Host: {host}, GPU: {gpu_index}, Words: {word_count}, "
                    f"Duration: {processing_time:.2f}s, WPS: {wps:.2f}"
                )
                
            except Exception as e:
                log_message(f"Error on host {host}: {str(e)}")
                
            # Mark task as done
            task_queue.task_done()
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            log_message(f"Worker error: {str(e)}")
            
    log_message(f"Worker for {host} (GPU {gpu_index}) shutting down")

async def process_prompt_with_context(prompt, host, model, system_message, prompt_id=None, prompt_file_modified_time=None, output_dir="responses"):
    """Process a single prompt with context fetching."""
    # Check if we need to regenerate the response
    if prompt_id and prompt_file_modified_time:
        if not should_regenerate(prompt_id, output_dir, prompt_file_modified_time):
            log_message(f"Skipping {prompt_id} - prompt unchanged since last run")
            return None, None
    
    # Extract URLs from prompt if present
    urls = extract_markdown_links(prompt)
    
    # Fetch context if URLs are present
    context_block = None
    if urls:
        url_list = [url for _, url in urls]
        contexts = await fetch_all_contexts(url_list)
        if any(contexts):
            context_block = create_context_block(contexts)
    
    # Get response from model
    response_text, start_time, duration, length = await chat(prompt, host, model, system_message, context_block)
    
    # Save response with timing information
    return save_response(
        prompt, 
        response_text, 
        output_dir, 
        prompt_id, 
        start_time, 
        duration, 
        length
    )

async def main(config_path, prompts_path, output_dir):
    """Main function to process prompts using Ollama."""
    try:
        # Load configuration
        config = load_config(config_path)
        system_msg = config.get("system_message", "")
        model = config["model"]
        
        # Load prompts from both specified path and prompts directory
        prompts = load_prompts(prompts_dir="prompts", prompts_files=[prompts_path] if prompts_path else None)
        log_message(f"Total prompts to process: {len(prompts)}")
        
        # Create directory for responses
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a task queue
        task_queue = asyncio.Queue()
        
        # Add tasks to the queue
        for prompt in prompts:
            # Format: [prompt_id, prompt_content, system_message, modified_time]
            if isinstance(prompt, dict):
                prompt_id = prompt.get('id', None)
                prompt_content = prompt.get('content', prompt.get('prompt', ''))
                modified_time = prompt.get('_modified_time', datetime.now())
                task_queue.put_nowait((prompt_id, prompt_content, system_msg, modified_time))
            else:
                # If prompt is a string (simple format)
                task_queue.put_nowait((None, prompt, system_msg, datetime.now()))
        
        # Create a list of worker tasks, one for each Ollama instance
        tasks = []
        for host, gpu_index in config.get("ollama_instances", {}).items():
            worker_task = asyncio.create_task(
                worker(host, gpu_index, model, task_queue, output_dir)
            )
            tasks.append(worker_task)
        
        # Add sentinel values to stop workers
        for _ in range(len(tasks)):
            await task_queue.put(None)
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks)
        
    except asyncio.CancelledError:
        log_message("Process interrupted by user. Exiting...")
    except Exception as e:
        log_message(f"Error: {str(e)}")
        traceback.print_exc()

def load_config(config_path):
    """Load the configuration from a TOML file."""
    try:
        return toml.load(config_path)
    except Exception as e:
        log_message(f"Error loading config: {str(e)}")
        raise

def load_prompts(prompts_dir="prompts", prompts_files=None):
    """
    Load prompts from both a directory and specific JSONL/JSON files.
    Tracks modification times for each prompt source file.
    """
    prompts = []
    
    # Process specific JSONL/JSON files if provided
    if prompts_files:
        for file_path in prompts_files:
            path = Path(file_path)
            if path.exists() and path.is_file():
                modified_time = datetime.fromtimestamp(path.stat().st_mtime)
                log_message(f"Loading prompts from file: {path}")
                
                if path.suffix.lower() == '.json':
                    with open(path, "r", encoding="utf-8") as file:
                        data = json.load(file)
                        # Add source file and modification time to each prompt
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict):
                                    item['_source_file'] = str(path)
                                    item['_modified_time'] = modified_time
                            prompts.extend(data)
                        else:
                            if isinstance(data, dict):
                                data['_source_file'] = str(path)
                                data['_modified_time'] = modified_time
                            prompts.append(data)
                
                elif path.suffix.lower() == '.jsonl':
                    with open(path, "r", encoding="utf-8") as file:
                        for line in file:
                            if line.strip():
                                try:
                                    item = json.loads(line)
                                    if isinstance(item, dict):
                                        item['_source_file'] = str(path)
                                        item['_modified_time'] = modified_time
                                    prompts.append(item)
                                except json.JSONDecodeError:
                                    log_message(f"Error parsing JSONL line in {path}")
    
    # Process prompts directory
    if prompts_dir:
        prompts_path = Path(prompts_dir)
        if prompts_path.exists() and prompts_path.is_dir():
            log_message(f"Loading prompts from directory: {prompts_path}")
            for json_file in prompts_path.glob("*.json"):
                modified_time = datetime.fromtimestamp(json_file.stat().st_mtime)
                
                with open(json_file, "r", encoding="utf-8") as file:
                    try:
                        data = json.load(file)
                        # Add source file and modification time to each prompt
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict):
                                    item['_source_file'] = str(json_file)
                                    item['_modified_time'] = modified_time
                            prompts.extend(data)
                        else:
                            if isinstance(data, dict):
                                data['_source_file'] = str(json_file)
                                data['_modified_time'] = modified_time
                            prompts.append(data)
                    except json.JSONDecodeError:
                        log_message(f"Error parsing JSON file: {json_file}")
    
    if not prompts:
        raise ValueError("No prompts found in specified paths")
    
    log_message(f"Loaded {len(prompts)} prompts in total")
    return prompts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ollama Batch Processing Client")
    parser.add_argument("--config", type=str, default="config.toml", help="Path to the configuration TOML file")
    parser.add_argument("--prompts", type=str, help="Path to additional JSON/JSONL file with prompts (will also load from prompts directory)")
    parser.add_argument("--output_dir", type=str, default="responses", help="Directory to save the response JSON files")

    args = parser.parse_args()

    try:
        asyncio.run(main(args.config, args.prompts, args.output_dir))
        log_message("All prompts processed. Exiting...")
    except KeyboardInterrupt:
        log_message("Process interrupted by user. Exiting...")

