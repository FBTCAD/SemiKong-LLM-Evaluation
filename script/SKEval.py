import json
import requests
import time
import logging
import os
import torch
from sentence_transformers import SentenceTransformer, util
from fuzzywuzzy import fuzz
from rapidfuzz import fuzz as rapid_fuzz

# Suppress Hugging Face logging and progress bars
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# Set up logging for your script
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Embedded configuration
CONFIG = {
    "api_url": "http://192.168.18.12:1234/v1/chat/completions",
    "model_name": "semikong-70b",
    "max_tokens": 150,
    "temperature": 0.3,
    "fuzzy_threshold": 60,  # Adjust this threshold as needed
    "keyword_threshold": 0.6,  # Threshold for keyword-based matching
}

# Hardcoded file paths
TEST_CASES_FILE = "edatcad.json"
RESULTS_FILE = "SK11edatcad_short.txt"

# Load a pre-trained sentence transformer model for semantic similarity (not used in this version)
SIMILARITY_MODEL = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# Function to normalize text by removing hyphens and extra spaces
def normalize_text(text):
    """
    Normalize text by removing hyphens and extra spaces.
    """
    # Replace hyphens with spaces
    text = text.replace("-", " ")
    # Remove extra spaces
    text = " ".join(text.split())
    return text

# Function to query the LM Studio API
def query_llm(input_text, api_url, model_name, max_tokens, temperature):
    # Append instructions for a shorter answer
    input_text += " Provide the closest shorter answer."
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": input_text}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        response = requests.post(api_url, headers={"Content-Type": "application/json"}, json=payload)
        response.raise_for_status()
        llm_response = response.json()["choices"][0]["message"]["content"].strip()
        return llm_response
    except requests.exceptions.RequestException as e:
        logger.error(f"Error querying LM Studio API: {e}")
        return None

# Function to check if the expected output matches the LLM's response using fuzzy matching (EvalSK6)
def fuzzy_match(expected_output, llm_response, fuzzy_threshold):
    expected_output = normalize_text(expected_output.lower())
    llm_response = normalize_text(llm_response.lower())
    similarity_score = fuzz.partial_ratio(expected_output, llm_response)
    return similarity_score >= fuzzy_threshold, similarity_score

# Function to check if the expected output matches the LLM's response using keyword-based matching (EvalSK7)
def keyword_match(expected_output, llm_response):
    expected_output = normalize_text(expected_output.lower())
    llm_response = normalize_text(llm_response.lower())
    keywords = expected_output.split()
    match = all(keyword in llm_response for keyword in keywords)
    similarity_score = sum(keyword in llm_response for keyword in keywords) / len(keywords)
    return match, similarity_score * 100

# Function to check if the expected output matches the LLM's response using rapidfuzz (EvalSK8)
def rapidfuzz_match(expected_output, llm_response):
    expected_output = normalize_text(expected_output.lower())
    llm_response = normalize_text(llm_response.lower())
    similarity_score = rapid_fuzz.partial_ratio(expected_output, llm_response)
    return similarity_score >= CONFIG["fuzzy_threshold"], similarity_score

# Main function to run the evaluation
def evaluate_llm(test_cases, api_url, model_name, max_tokens, temperature):
    results = []
    for case in test_cases:
        logger.info(f"Processing Test Case {case['id']}: {case['input']}")
        llm_response = query_llm(case["input"], api_url, model_name, max_tokens, temperature)
        
        if llm_response is None:
            logger.warning(f"Skipping Test Case {case['id']} due to API error.")
            results.append({
                "id": case["id"],
                "category": case["category"],
                "input": case["input"],
                "expected_output": case["expected_output"],
                "llm_response": "API Error",
                "fuzzy_match": False,
                "keyword_match": False,
                "rapidfuzz_match": False,
                "fuzzy_score": 0,
                "keyword_score": 0,
                "rapidfuzz_score": 0,
                "average_score": 0,
                "result": "Fail",
            })
            continue

        # Check matches using all three methods
        fuzzy_result, fuzzy_score = fuzzy_match(case["expected_output"], llm_response, CONFIG["fuzzy_threshold"])
        keyword_result, keyword_score = keyword_match(case["expected_output"], llm_response)
        rapidfuzz_result, rapidfuzz_score = rapidfuzz_match(case["expected_output"], llm_response)

        # Calculate the average score
        average_score = (fuzzy_score + keyword_score + rapidfuzz_score) / 3

        # Determine pass/fail based on the average score and threshold
        result = "Pass" if average_score >= CONFIG["fuzzy_threshold"] else "Fail"

        results.append({
            "id": case["id"],
            "category": case["category"],
            "input": case["input"],
            "expected_output": case["expected_output"],
            "llm_response": llm_response,
            "fuzzy_match": fuzzy_result,
            "keyword_match": keyword_result,
            "rapidfuzz_match": rapidfuzz_result,
            "fuzzy_score": fuzzy_score,
            "keyword_score": keyword_score,
            "rapidfuzz_score": rapidfuzz_score,
            "average_score": average_score,
            "result": result,
        })

    return results

# Function to write results to a file
def write_results(results, output_file):
    with open(output_file, "w", encoding="utf-8") as file:
        for result in results:
            file.write(f"=== Test Case {result['id']} ===\n")
            file.write(f"Category: {result['category']}\n")
            file.write(f"Input: {result['input']}\n")
            file.write(f"Expected: {result['expected_output']}\n")
            file.write(f"LLM Response: {result['llm_response']}\n")
            file.write(f"Fuzzy Match: {'Pass' if result['fuzzy_match'] else 'Fail'} (Score: {result['fuzzy_score']:.2f})\n")
            file.write(f"Keyword Match: {'Pass' if result['keyword_match'] else 'Fail'} (Score: {result['keyword_score']:.2f})\n")
            file.write(f"RapidFuzz Match: {'Pass' if result['rapidfuzz_match'] else 'Fail'} (Score: {result['rapidfuzz_score']:.2f})\n")
            file.write(f"Average Score: {result['average_score']:.2f}\n")
            file.write(f"Result: {result['result']}\n\n")

        # Calculate accuracy
        total_cases = len(results)
        passed_cases = sum(1 for result in results if result["result"] == "Pass")
        accuracy = (passed_cases / total_cases) * 100
        file.write(f"Accuracy: {accuracy:.2f}%\n")

# Function to load test cases from the hardcoded JSON file
def load_test_cases():
    try:
        with open(TEST_CASES_FILE, "r") as file:
            test_cases = json.load(file)["test_cases"]
        return test_cases
    except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
        logger.error(f"Error loading test cases from {TEST_CASES_FILE}: {e}")
        exit(1)

# Main execution
if __name__ == "__main__":
    # Load test cases from the hardcoded JSON file
    test_cases = load_test_cases()

    # Record the start time
    start_time = time.time()

    # Evaluate the LLM using embedded configuration and hardcoded test cases
    results = evaluate_llm(
        test_cases,
        api_url=CONFIG["api_url"],
        model_name=CONFIG["model_name"],
        max_tokens=CONFIG["max_tokens"],
        temperature=CONFIG["temperature"],
    )

    # Write results to the hardcoded output file
    write_results(results, RESULTS_FILE)
    logger.info(f"Results written to {RESULTS_FILE}")

    # Record the stop time and calculate execution time
    stop_time = time.time()
    execution_time = stop_time - start_time
    logger.info(f"Total execution time: {execution_time:.2f} seconds")