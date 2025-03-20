from magika import Magika
import re
import json
import urllib.parse
import logging
import os
from datetime import datetime

# Set up logging configuration
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"language_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(levelname)s - %(message)s',
  filename=log_file,
)
logger = logging.getLogger(__name__)

def detect_language(snippet):
  """
  Detects the programming language of a code snippet using Magika.

  Args:
  snippet (str): The code snippet to analyze

  Returns:
  str: The detected programming language or "Unknown" if not detected
  """
  try:
    m = Magika()
    result = m.identify_bytes(snippet.encode('utf-8'))
    return result.output.label
  except Exception as e:
    # logger.error(f"Error detecting language: {e}")
    return "Unknown"

def preprocess(payload):
  """
  Performs URL decoding on the input payload.

  Args:
  payload (str): The URL-encoded string to decode

  Returns:
  str: The decoded string
  """
  try:
    urldecoded = urllib.parse.unquote_plus(payload)
    # Look for content inside script tags
    script_pattern = re.compile(r'<script[^>]*>.*?</script>', re.DOTALL | re.IGNORECASE)
    script_match = script_pattern.search(urldecoded)

    if script_match:
      script_content = script_match.group(1)
      return script_content.strip()
    else:
      return urldecoded

  except Exception as e:
    # logger.error(f"Error decoding URL: {e}")
    return payload


if __name__ == "__main__":
  # Load the JSON file
  try:
    logger.info("Starting language detection process")
    with open('classified_requests.json', 'r', encoding='utf-8') as file:
      data = json.load(file)
    logger.info(f"Loaded JSON with {len(data)} records")
    
    # Process each record
    processed_count = 0
    for item in data:
      if 'payload' not in item:
        logger.warning(f"'payload' field not found in item: {item}")
        continue
        
      payload = item['payload']
      processed_payload = preprocess(payload)
      language = detect_language(processed_payload)
      
      # Add the detected language to the item
      item['detected_language'] = language
      
      processed_count += 1
      if processed_count % 100 == 0:
        logger.info(f"Processed {processed_count}/{len(data)} records")
    
    # Save the results to a new JSON file
    with open('data_with_languages.json', 'w', encoding='utf-8') as outfile:
      json.dump(data, outfile, indent=4)
    logger.info("Processing complete! Results saved to 'data_with_languages.json'")
    
  except Exception as e:
    logger.error(f"An error occurred: {e}", exc_info=True)
