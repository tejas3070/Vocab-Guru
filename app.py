#Import necessary modules
from flask import Flask, render_template, request, redirect, url_for
import torch
from nltk.corpus import wordnet
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai
from transformers import BitsAndBytesConfig
import os
import re
import concurrent.futures

app = Flask(__name__)

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
openai.api_key = "<YOUR-API-KEY-HERE>"

# BitsAndBytes quantization configuration to ensure lower memory usage
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Loading fine-tuned model and tokenizer
finetuned_model_path = "./finetuned_llama2"
tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    finetuned_model_path,
    quantization_config=quant_config,
    device_map="auto"
).to("cuda" if torch.cuda.is_available() else "cpu")

#Clean the Fine-tuned LLM response to obtain only the improved sentence
def extract_improved_sentence(output, original_prompt):
    # Remove the original prompt from the output
    response = output.replace(original_prompt, "").strip()

    # Remove repetitive or irrelevant symbols using regex
    clean_response = re.sub(r"[=]{2,}", "", response)  # Remove repeated '=' symbols
    clean_response = re.sub(r"\n+", " ", clean_response).strip()  # Replace newlines with spaces
    clean_response = re.sub(r"\s+", " ", clean_response).strip()  # Remove extra spaces

    # Return the first meaningful sentence
    return clean_response.split(".")[0] + "." if "." in clean_response else clean_response

# Function for extracting sysnsets
def extract_synonyms(sentence):
    words = sentence.split()
    synonyms = {}
    for word in words:
        synsets = wordnet.synsets(word)
        syn_list = set(lemma.name() for syn in synsets for lemma in syn.lemmas() if lemma.name() != word)
        if syn_list:
            synonyms[word] = list(syn_list)
    return synonyms

#Function with lower output token limit for Obtaining results from Fine-tuned LLM
def prompt_finetuned_model_with_postprocessing(prompt, max_new_tokens=50, temperature=0.7):
    
	# Defining the system prompt with few shot prompting strategy
    system_prompt = (
    "You are a helpful assistant that improves sentences with advanced vocabulary.\n###\n"
    "Here are examples of your task:\n\n"
    "Example 1:\n"
    "Input: I will try to do my best and make a change in the team.\n"
    "Output: I will strive to maximize my potential and instigate transformation within the team.\n\n"
    "Example 2:\n"
    "Input: The cat is on the mat.\n"
    "Output: The feline rests upon the carpet.\n\n"
    "Now improve this sentence:\n"
    "Input: {user_input}\n###\n"
    "Output:"
    )
	
    # Combine system prompt and user input
    final_prompt = system_prompt.format(user_input=prompt)
    inputs = tokenizer(final_prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
    #Generate using the fine-tuned LLM
    outputs = model.generate(
    inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=max_new_tokens,
    temperature=temperature,
    no_repeat_ngram_size=3,
    pad_token_id=tokenizer.eos_token_id
    )
    
    #Obtain the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "###" in response:
       response = response.split("###")[-1].strip()
    improved_sentence = extract_improved_sentence(response, prompt)
    return improved_sentence

#Function with higher output token limit for obtaining results from Fine-tuned LLM
def prompt_finetuned_model_with_postprocessing_2(prompt, max_new_tokens=80, temperature=0.7):
    
	# Defining the system prompt with few shot prompting strategy
    system_prompt = (
    "You are a helpful assistant that improves sentences with advanced vocabulary.\n###\n"
    "Here are examples of your task:\n\n"
    "Example 1:\n"
    "Input: I will try to do my best and make a change in the team.\n"
    "Output: I will strive to maximize my potential and instigate transformation within the team.\n\n"
    "Example 2:\n"
    "Input: The cat is on the mat.\n"
    "Output: The feline rests upon the carpet.\n\n"
    "Now improve this sentence:\n"
    "Input: {user_input}\n###\n"
    "Output:"
    )

    # Combine system prompt and user input
    final_prompt = system_prompt.format(user_input=prompt)
    inputs = tokenizer(final_prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
    
    #Generate using the fine-tuned LLM
    outputs = model.generate(
    inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=max_new_tokens,
    temperature=temperature,
    no_repeat_ngram_size=3,
    pad_token_id=tokenizer.eos_token_id
    )
	
    #Obtain the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "###" in response:
       response = response.split("###")[-1].strip()
    improved_sentence = extract_improved_sentence(response, prompt)
    return improved_sentence

#Function to Prompt-GPT4
def prompt_gpt4(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"].strip()


# Function to extract better equivalents of words
def get_better_equivalents(text):

    """Generates better equivalents for each word in the text."""
    words = text.split()  # Split the input text into words
    word_suggestions = []
    index = 1  # Used to number the output suggestions

    for word in words:
        synonyms = set()
        # Get synonyms from WordNet for each word
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())

        # Filter out identical words and irrelevant synonyms
        relevant_synonyms = {syn for syn in synonyms if syn.lower() != word.lower() and len(syn) > 2}

        # If synonyms are found, filter them based on context
        if relevant_synonyms:
            filtered_synonyms = filter_synonyms_by_context(word, list(relevant_synonyms)[:10], text)
            if filtered_synonyms:
                # Format the output as '1. word == synonym1, synonym2'
                word_suggestions.append(f"{index}. {word.capitalize()} == {', '.join(filtered_synonyms[:5])}")
                index += 1

    return "\n".join(word_suggestions)


# Function to filter synonyms using GPT-4
def filter_synonyms_by_context(word, synonyms, sentence):
    """Filters synonyms based on context using GPT-4."""
    system_prompt = (
    "You are an expert in contextual word replacement.\n"
    "Here are examples of your task:\n\n"
    "Example 1:\n"
    "Input: I will try to do my best and make a change in the team.\n"
    "Output: do my best can be replaced with maximize my potential, change can be replaced with difference.\n\n"
    "Example 2:\n"
    "Input: The cat is on the mat.\n"
    "Output: cat can be replaced with feline, mat can be replaced with carpet.\n\n"
    "Now improve this sentence:\n"
    "Input: {user_input}\n###\n"
    "Output:"
    )

    
    prompt = (
        f"Here is a sentence: '{sentence}'. "
        f"Which of these words can replace '{word}' in the sentence without changing the meaning and while remaining contextually correct? "
        f"Provide only a comma-separated list of relevant synonyms: {', '.join(synonyms)}."
    )

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt },
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.7
    )

    # Extract GPT-4's response and clean the result
    filtered_response = response['choices'][0]['message']['content'].strip()
    filtered_synonyms = [syn.strip() for syn in filtered_response.split(",") if len(syn.strip()) > 0]

    return filtered_synonyms

#Function that calls LLMs 
def process_sentence(sentence):
    
    better_equivalents = get_better_equivalents(sentence)
    # Construct user prompts
    prompt_1 = f"Improve this sentence: {sentence}"
    prompt_2 = f"Improve this sentence: {sentence}\n"f"Use these suggested words if contextually relevant:\n"f"Options:\n{better_equivalents}"
    prompt_gpt = f"Improve this sentence with advanced vocabulary while maintaining context: {sentence}"
    
    response_1 = prompt_finetuned_model_with_postprocessing(prompt_1)
    response_2 = prompt_finetuned_model_with_postprocessing_2(prompt_2)
    response_3 = prompt_gpt4(prompt_gpt)
    return response_1, response_2, response_3, better_equivalents


#Flask Front-end implementation
@app.route("/", methods=["GET", "POST"])
def input_page():
    """Handles the input page where the user submits a sentence."""
    if request.method == "POST":
        sentence = request.form["sentence"]
        return redirect(url_for("results_page", sentence=sentence))
    return render_template("input.html")


@app.route("/results")
def results_page():
    """Displays the results page with improved sentences and better equivalents."""
    sentence = request.args.get("sentence")
    response_1, response_2, response_3, better_equivalents = process_sentence(sentence)

    return render_template(
        "results.html",
        original_sentence=sentence,
        fine_tuned_llm=response_1,
        fine_tuned_llm_synsets=response_2,
        gpt4=response_3,
        better_equivalents=better_equivalents,
    )


if __name__ == "__main__":
    app.run(debug=True)
    #app.run(host="0.0.0.0", port=5050, debug=True)

