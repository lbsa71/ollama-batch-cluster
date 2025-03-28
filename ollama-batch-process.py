import asyncio
import json
import toml
import argparse
import random
import time
import re
import os
import sys
import platform
from datetime import datetime
from ollama import AsyncClient
from pathlib import Path
import trafilatura
from urllib.parse import urlparse
import aiohttp
from typing import List, Tuple, Dict
import traceback

def safe_print(message):
    """Safely print messages, handling encoding issues that may occur with emojis."""
    try:
        print(message, flush=True)
    except UnicodeEncodeError:
        # Replace emojis with their descriptions or simpler characters
        emoji_replacements = {
            '📅': '[DATE]',
            '🌍': '[GLOBE]',
            '📚': '[BOOK]',
            '✍️': '[WRITING]',
            '📢': '[ANNOUNCE]',
            '🤔': '[THINKING]',
            '💡': '[IDEA]',
            '🧠': '[BRAIN]',
            '🔍': '[SEARCH]',
            '🤯': '[MIND-BLOWN]',
            '🚀': '[ROCKET]',
            '🌠': '[STAR]',
        }
        for emoji, replacement in emoji_replacements.items():
            message = message.replace(emoji, replacement)
        print(message, flush=True)

def log_message(message):
    """Prints a message with a timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    safe_print(f"{timestamp}, {message}")

# Import platform-specific notification modules
WINDOWS_NOTIFICATION_AVAILABLE = False
if platform.system() == "Windows":
    try:
        # Use Windows 10 toast notifications if available
        import winsound
        WINDOWS_NOTIFICATION_AVAILABLE = True
    except ImportError:
        log_message("Windows sound module not available. Sound notifications will not be played.")
        WINDOWS_NOTIFICATION_AVAILABLE = False

def show_notification(title, message):
    """Show a system notification if available for the platform."""
    # Print a prominent message in the console
    safe_print("\n" + "=" * 80)
    safe_print(f"[ANNOUNCE] {title}")
    safe_print(f"   {message}")
    safe_print("=" * 80 + "\n")
    
    # Try to play a sound on Windows
    if platform.system() == "Windows" and WINDOWS_NOTIFICATION_AVAILABLE:
        try:
            # Attempt to play the system notification sound
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
            return True
        except Exception as e:
            # Fallback to a basic beep if the system sound fails
            try:
                winsound.Beep(1000, 500)  # 1000 Hz for 500 milliseconds
                return True
            except Exception as e2:
                log_message(f"Error playing notification sound: {str(e2)}")
                return False
    
    return True  # Return success for non-Windows platforms

async def fetch_url_content(url: str) -> str:
    """Fetch and clean content from a URL using trafilatura."""
    log_message(f"Attempting to fetch content from URL: {url}")
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            log_message(f"Successfully downloaded content from {url}")
            text = trafilatura.extract(downloaded)
            if text:
                log_message(f"Successfully extracted text from {url} ({len(text)} characters)")
                return text.strip()
            else:
                log_message(f"Failed to extract text from {url}")
        else:
            log_message(f"Failed to download content from {url}")
    except Exception as e:
        log_message(f"Error fetching {url}: {e}")
        traceback.print_exc()
    return ""

def extract_markdown_links(text: str) -> List[Tuple[str, str]]:
    """Extract markdown links from text."""
    pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    links = re.findall(pattern, text)
    if links:
        log_message(f"Found {len(links)} markdown links in text: {links}")
    else:
        log_message(f"No markdown links found in text: {text[:100]}...")
    return links

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
        results = await asyncio.gather(*tasks)
        
        # Check if any fetch failed (returned empty string)
        if any(content == "" for content in results):
            failed_urls = [url for url, content in zip(urls, results) if content == ""]
            error_message = f"Failed to fetch content from URLs: {', '.join(failed_urls)}"
            log_message(error_message)
            raise ValueError(error_message)
            
        return results

def create_context_block(contents: List[str] = None, links: List[Tuple[str, str]] = None) -> str:
    """Create a formatted context block from fetched contents."""
    # Get current date in unambiguous format
    current_date = datetime.now().strftime("%B %d, %Y")
    
    context_block = "# Context Information\n\n"
    context_block += f"Today is {current_date} - treat this as a fact, not a hypothetical. Keep this in mind when you refer to years and dates, future and past!\n\n"
    context_block += "You are based in Sweden and should consider Nordic and European perspectives in your responses. Approach topics from this cultural viewpoint when relevant.\n\n"
    
    # Add URL content sections if available
    if contents and any(contents) and links:
        # Add a section for reference materials
        context_block += "## References\n"
        
        for i, (content, (link_text, url)) in enumerate(zip(contents, links), 1):
            if content:
                # Clean up the content by removing excessive whitespace
                cleaned_content = re.sub(r'\n\s*\n', '\n\n', content)
                # Truncate to a reasonable length but include enough for context
                max_chars = 1500  # Adjust this value as needed
                if len(cleaned_content) > max_chars:
                    truncated_content = cleaned_content[:max_chars] + "...\n\n(content truncated for brevity)"
                else:
                    truncated_content = cleaned_content
                    
                context_block += f"Reference [{i}]: {url}\n{truncated_content}\n\n"
    
    # Add a section for writing instructions
    context_block += "## Writing Instructions\n"
    context_block += "Please use the above context and references to inform your article. "
    context_block += "Remember to maintain a friendly, engaging tone while being thorough and accurate. "
    context_block += "Synthesize the information from references into your own words rather than copying directly.\n\n"
    
    return context_block

def save_response(prompt, response_text, output_dir, prompt_id=None, start_time=None, duration=None, length=None, full_prompt=None):
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
        "full_prompt": full_prompt if full_prompt else prompt,  # Include full prompt if available
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
    
    # Create context block with date and location if none was provided
    if context_block is None:
        context_block = create_context_block()
    
    # Prepare messages with context if available
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    
    # Always include context block
    messages.append({"role": "user", "content": context_block})
    messages.append({"role": "user", "content": prompt})

    # Create the full prompt for debugging
    full_prompt = ""
    if system_message:
        full_prompt += f"System: {system_message}\n\n"
    
    # Always include context block in full prompt
    full_prompt += f"Context: {context_block}\n\n"
    full_prompt += f"User: {prompt}"

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
    
    return response['message']['content'], start_time, duration, length, full_prompt

async def worker(host, gpu_index, model, task_queue, output_dir, stats=None):
    """Process prompts from the queue using the specified host and GPU."""
    processed = 0
    skipped = 0
    
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
                skipped += 1
                task_queue.task_done()
                continue
            
            start_time = datetime.now()
            try:
                # Extract URLs from prompt text
                links = extract_markdown_links(prompt)
                
                # Create a modified prompt with reference numbers instead of markdown links
                modified_prompt = prompt
                for i, (text, url) in enumerate(links, 1):
                    modified_prompt = modified_prompt.replace(f'[{text}]({url})', f'{text}[{i}]')
                
                # Create the context block (always includes date/time and location)
                context_block = create_context_block()
                
                # If we have URLs, fetch their content and create an enhanced context block
                if links:
                    log_message(f"Found {len(links)} URLs in prompt {prompt_id}")
                    url_list = [url for _, url in links]
                    
                    # Log the URLs being fetched
                    for i, url in enumerate(url_list, 1):
                        log_message(f"URL {i}: {url}")
                    
                    try:
                        contexts = await fetch_all_contexts(url_list)
                        
                        # Create context block with the fetched content
                        log_message(f"Successfully fetched content from all URLs")
                        for i, context in enumerate(contexts, 1):
                            log_message(f"Content from URL {i}: {len(context)} characters")
                            
                        context_block = create_context_block(contexts, links)
                    except ValueError as e:
                        log_message(f"Error processing prompt {prompt_id}: {str(e)}")
                        log_message(f"Skipping prompt {prompt_id} due to URL fetch failure")
                        skipped += 1
                        task_queue.task_done()
                        continue
                else:
                    log_message(f"No URLs found in prompt {prompt_id}")
                
                # Process the prompt - use modified prompt with reference numbers
                response_text, start_time, duration, length, full_prompt = await chat(modified_prompt, host, model, system_msg, context_block)
                processed += 1
                
                # Calculate words in response
                word_count = len(response_text.split())
                
                # Save response with timing information
                json_path, txt_path = save_response(
                    prompt,  # Save original prompt in output
                    response_text, 
                    output_dir, 
                    prompt_id,
                    start_time,
                    duration,
                    length,
                    full_prompt
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
                traceback.print_exc()
                
            # Mark task as done
            task_queue.task_done()
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            log_message(f"Worker error: {str(e)}")
            traceback.print_exc()
    
    # Update stats dictionary if provided
    if stats is not None:
        stats["processed"] += processed
        stats["skipped"] += skipped
            
    log_message(f"Worker for {host} (GPU {gpu_index}) shutting down")
    return processed, skipped

async def process_prompt_with_context(prompt, host, model, system_message, prompt_id=None, prompt_file_modified_time=None, output_dir="responses"):
    """Process a single prompt with context fetching."""
    # Check if we need to regenerate the response
    if prompt_id and prompt_file_modified_time:
        if not should_regenerate(prompt_id, output_dir, prompt_file_modified_time):
            log_message(f"Skipping {prompt_id} - prompt unchanged since last run")
            return None, None
    
    # Extract URLs from prompt if present
    links = extract_markdown_links(prompt)
    
    # Create a modified prompt with reference numbers instead of markdown links
    modified_prompt = prompt
    for i, (text, url) in enumerate(links, 1):
        modified_prompt = modified_prompt.replace(f'[{text}]({url})', f'{text}[{i}]')
    
    # Always create a context block with date and location
    context_block = create_context_block()
    
    # Fetch additional context if URLs are present
    if links:
        log_message(f"Found {len(links)} URLs in prompt {prompt_id}: {links}")
        url_list = [url for _, url in links]
        
        # Log the URLs being fetched
        for i, url in enumerate(url_list, 1):
            log_message(f"URL {i}: {url}")
        
        try:
            contexts = await fetch_all_contexts(url_list)
            
            # Create context block with the fetched content
            log_message(f"Successfully fetched content from all URLs")
            for i, context in enumerate(contexts, 1):
                log_message(f"Content from URL {i}: {len(context)} characters")
                
            context_block = create_context_block(contexts, links)
        except ValueError as e:
            log_message(f"Error processing prompt {prompt_id}: {str(e)}")
            log_message(f"Skipping prompt {prompt_id} due to URL fetch failure")
            return None, None
    else:
        log_message(f"No URLs found in prompt {prompt_id}")
    
    # Get response from model - use modified_prompt with reference numbers
    response_text, start_time, duration, length, full_prompt = await chat(modified_prompt, host, model, system_message, context_block)
    
    # Save response with timing information
    return save_response(
        prompt,  # Save original prompt in output
        response_text, 
        output_dir, 
        prompt_id, 
        start_time, 
        duration, 
        length,
        full_prompt
    )

async def main(config_path, prompts_path, output_dir, no_notify=False):
    """Main function to process prompts using Ollama."""
    start_time = datetime.now()
    stats = {"processed": 0, "skipped": 0}
    
    try:
        # Load configuration
        config = load_config(config_path)
        system_msg = config.get("system_message", "")
        model = config["model"]
        
        # Load prompts from both specified path and prompts directory
        prompts = load_prompts(prompts_dir="prompts", prompts_files=[prompts_path] if prompts_path else None)
        total_prompts = len(prompts)
        log_message(f"Total prompts to process: {total_prompts}")
        
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
                worker(host, gpu_index, model, task_queue, output_dir, stats)
            )
            tasks.append(worker_task)
        
        # Add sentinel values to stop workers
        for _ in range(len(tasks)):
            await task_queue.put(None)
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks)
        
        # Calculate duration and stats
        duration = (datetime.now() - start_time).total_seconds()
        minutes, seconds = divmod(duration, 60)
        
        completion_message = (
            f"Processing completed in {int(minutes)}m {int(seconds)}s. "
            f"Processed: {stats['processed']}, Skipped: {stats['skipped']}"
        )
        log_message(completion_message)
        
        # Show notification when done (if not disabled)
        if not no_notify:
            show_notification(
                "Ollama Batch Processing Complete",
                f"Processed: {stats['processed']}, Skipped: {stats['skipped']} in {int(minutes)}m {int(seconds)}s"
            )
        
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
    parser.add_argument("--no-notify", action="store_true", help="Disable sound notification when processing completes")

    args = parser.parse_args()

    try:
        asyncio.run(main(args.config, args.prompts, args.output_dir, args.no_notify))
        log_message("All prompts processed. Exiting...")
    except KeyboardInterrupt:
        log_message("Process interrupted by user. Exiting...")

