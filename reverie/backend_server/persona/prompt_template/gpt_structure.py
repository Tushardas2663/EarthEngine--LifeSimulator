"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: gpt_structure.py
Description: Wrapper functions for calling LLM APIs.
MODIFIED FOR HACKATHON:
- Now supports OpenAI, self-hosted Ollama, and Google Gemini.
- All requests are standardized through a single function.
- Includes an API call counter and robust error/rate-limit handling.
"""
import json
import openai
import time 

from utils import *

# --- HACKATHON CHANGE 1: CHOOSE YOUR LLM PROVIDER ---
USE_OPENAI = False
USE_OLLAMA = True # Set this to True to use your self-hosted LLM
USE_GEMINI = False 

# --- HACKATHON CHANGE 2: API & MODEL CONFIGURATION ---
OPENAI_MODEL = "gpt-4o-mini"
OLLAMA_URL = "https://4457e1f1fb76.ngrok-free.app/" # IMPORTANT: PASTE YOUR NGROK URL HERE
OLLAMA_MODEL = 'llama3:8b'
GEMINI_MODEL = 'gemini-1.5-flash-latest'

# --- HACKATHON CHANGE 3: SETUP CLIENTS AND COUNTER ---
api_call_counter = 0
openai.api_key = openai_api_key

if USE_OLLAMA:
  import ollama
  # This header is crucial for bypassing the ngrok browser warning
  ollama_client = ollama.Client(
      host=OLLAMA_URL,
      headers={'ngrok-skip-browser-warning': 'any-value'}
  )
  print(f"--- Ollama Mode Enabled: Using model '{OLLAMA_MODEL}' at {OLLAMA_URL} ---")

if USE_GEMINI:
  import google.generativeai as genai
  genai.configure(api_key=gemini_api_key)
  gemini_model = genai.GenerativeModel(GEMINI_MODEL)
  print(f"--- Gemini Mode Enabled: Using model '{GEMINI_MODEL}' ---")


def ChatGPT_request(prompt): 
  """
  A single, unified function to make a request to the selected LLM server.
  """
  global api_call_counter
  api_call_counter += 1
  print(f"--- API Call #{api_call_counter} ---")

  try: 
    if USE_OLLAMA:
      response = ollama_client.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
      return response['message']['content']

    elif USE_GEMINI:
      response = gemini_model.generate_content(prompt)
      return response.text

    elif USE_OPENAI:
      completion = openai.ChatCompletion.create(model=OPENAI_MODEL, messages=[{"role": "user", "content": prompt}])
      return completion["choices"][0]["message"]["content"]
  
  except Exception as e: 
    # This block handles rate limits and other errors
    if "429" in str(e) and USE_GEMINI:
        print("--- Gemini Rate Limit Hit. Waiting 5 seconds before retrying... ---")
        time.sleep(5)
        try:
            print("--- Retrying API Call ---")
            response = gemini_model.generate_content(prompt)
            return response.text
        except Exception as retry_e:
            print(f"LLM REQUEST FAILED ON RETRY: {retry_e}")
            return f"LLM REQUEST FAILED: {retry_e}"
            
    print(f"LLM REQUEST FAILED: {e}")
    if "quota" in str(e).lower():
        return "LLM REQUEST FAILED: You exceeded your current quota."
    return f"LLM REQUEST FAILED: {e}"

def ChatGPT_safe_generate_response(prompt, 
                                     example_output,
                                     special_instruction,
                                     repeat=3,
                                     fail_safe_response="error",
                                     func_validate=None,
                                     func_clean_up=None,
                                     verbose=False): 
  prompt = '"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 
    try: 
      curr_gpt_response = ChatGPT_request(prompt).strip()
      # This handles cases where the LLM wraps the JSON in markdown
      if curr_gpt_response.startswith("```json"):
        curr_gpt_response = curr_gpt_response[7:-3].strip()
      
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]
      
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      
      if verbose: 
        print ("---- repeat count: \n", i, curr_gpt_response)
        print (curr_gpt_response)
        print ("~~~~")
    except Exception as e:
      print(f"Error processing response on attempt {i+1}: {e}")
      pass

  return False


def get_embedding(text, model="text-embedding-ada-002"):
  if not USE_OPENAI:
      print("WARNING: get_embedding requires OpenAI. Memory retrieval will be impaired.")
      return [0.0] * 1536 # Return a zero vector to prevent crashes

  text = text.replace("\n", " ")
  if not text: 
    text = "this is blank"
  try:
    result = openai.Embedding.create(input=[text], model=model)
    return result['data'][0]['embedding']
  except openai.error.OpenAIError as e:
    print("OpenAI Embedding Error:", e)
    return [0.0] * 1536

def generate_prompt(curr_input, prompt_lib_file): 
    if isinstance(curr_input, str): 
        curr_input = [curr_input]
    curr_input = [str(i) for i in curr_input]
    with open(prompt_lib_file, "r") as f:
        prompt = f.read()
    for count, i in enumerate(curr_input):   
        prompt = prompt.replace(f"!<INPUT {count}>!", i)
    if "<commentblockmarker>###</commentblockmarker>" in prompt: 
        prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
    return prompt.strip()


def safe_generate_response(prompt, 
                           gpt_parameter, # This is ignored
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False): 
    if verbose: 
        print (prompt)
    for i in range(repeat): 
        curr_gpt_response = ChatGPT_request(prompt)
        if func_validate(curr_gpt_response, prompt=prompt): 
            return func_clean_up(curr_gpt_response, prompt=prompt)
        if verbose: 
            print ("---- repeat count: ", i, curr_gpt_response)
            print (curr_gpt_response)
            print ("~~~~")
    return fail_safe_response