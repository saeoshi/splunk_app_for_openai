import sys
import json
import time
import os
import re
import requests
import splunklib.searchcommands as sc
from splunklib.searchcommands import Configuration, Option, StreamingCommand

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Get OpenAI API configuration from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

@Configuration()
class OpenAIStreaming(StreamingCommand):
    # Define `prompt` as an option for the search command
    prompt = Option(require=True)

    def stream(self, records):
        """
        Processes each record from the pipeline, substitutes placeholders in the prompt,
        and sends the processed prompt to OpenAI API.
        """
        # Check if API key is configured
        if not OPENAI_API_KEY:
            raise Exception("OPENAI_API_KEY environment variable is not set")

        # API request headers
        headers = {
            'Authorization': f'Bearer {OPENAI_API_KEY}',
            'Content-Type': 'application/json'
        }

        # Process records in batches to avoid token limits
        batch_size = 10  # Adjust based on token limits
        record_batch = []
        
        for record in records:
            record_batch.append(record)
            
            if len(record_batch) >= batch_size:
                yield from self._process_batch(record_batch, headers)
                record_batch = []
        
        # Process remaining records
        if record_batch:
            yield from self._process_batch(record_batch, headers)

    def _substitute_placeholders(self, prompt_template, record):
        """
        Substitute placeholders in the prompt template with values from the record.
        Placeholder format: {field_name}
        """
        # Find all placeholders in the format {field_name}
        placeholders = re.findall(r'\{([^}]+)\}', prompt_template)
        
        result_prompt = prompt_template
        for placeholder in placeholders:
            field_value = record.get(placeholder, f"[FIELD_NOT_FOUND:{placeholder}]")
            result_prompt = result_prompt.replace(f"{{{placeholder}}}", str(field_value))
        
        return result_prompt

    def _process_batch(self, records, headers):
        """
        Process a batch of records with OpenAI API calls.
        """
        for record in records:
            try:
                # Substitute placeholders in the prompt
                processed_prompt = self._substitute_placeholders(self.prompt, record)
                
                # API request payload
                data = {
                    'model': OPENAI_MODEL,
                    'messages': [
                        {
                            'role': 'user',
                            'content': processed_prompt
                        }
                    ],
                    'stream': True
                }

                # Send a POST request to the OpenAI API with streaming enabled
                response = requests.post(f"{OPENAI_API_BASE}/chat/completions",
                                         json=data, headers=headers, stream=True)

                # Set `_time` field as a UNIX timestamp
                event_time = time.time()

                # Process streaming response
                openai_response = ""
                
                # Parse streaming response line by line
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode('utf-8').strip()
                        
                        # Look for lines that start with "data: "
                        if line_text.startswith("data: "):
                            data_part = line_text[6:]  # Remove "data: " prefix
                            
                            # Check for stream termination
                            if data_part == "[DONE]":
                                break
                            
                            try:
                                # Try to parse the JSON data
                                chunk_data = json.loads(data_part)
                                
                                # Extract content from the chunk
                                if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                                    choice = chunk_data["choices"][0]
                                    if "delta" in choice and "content" in choice["delta"]:
                                        content = choice["delta"]["content"]
                                        openai_response += content
                                        
                            except json.JSONDecodeError:
                                # Skip malformed JSON lines
                                continue

                # Construct event for Splunk
                result = dict(record)  # Copy original record
                result.update({
                    "_time": event_time,
                    "original_prompt": self.prompt,
                    "processed_prompt": processed_prompt,
                    "status": "success" if response.status_code == 200 else "error",
                    "response_code": response.status_code,
                    "openai_response": openai_response,
                    "model": OPENAI_MODEL
                })

                if response.status_code != 200:
                    result.update({
                        "error_message": f"OpenAI API error: {response.status_code}"
                    })

                yield result

            except Exception as e:
                # Handle exceptions and return error message
                error_result = dict(record)  # Copy original record
                error_result.update({
                    "_time": time.time(),
                    "original_prompt": self.prompt,
                    "processed_prompt": self._substitute_placeholders(self.prompt, record),
                    "status": "exception",
                    "error_message": str(e),
                    "openai_response": "",
                    "model": OPENAI_MODEL
                })
                yield error_result

# Register the command in Splunk
if __name__ == "__main__":
    from splunklib.searchcommands import dispatch
    dispatch(OpenAIStreaming, sys.argv, sys.stdin, sys.stdout, __name__)
