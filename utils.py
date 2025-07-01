# utils.py
from ctransformers import AutoModelForCausalLM
import os
import time

# Global cache for models
model_cache = {}

def load_model(model_name="TheBloke/Mistral-7B-Instruct-v0.2-GGUF"):
    """
    Load a CTransformers model for text generation.
    
    Args:
        model_name (str): Name of the model to load from HuggingFace
        
    Returns:
        tuple: (None, model) - Just return the model, no tokenizer needed
    """
    global model_cache
    
    # If model already loaded, return from cache
    if model_name in model_cache:
        return None, model_cache[model_name]
        
    try:
        # Initialize the model from HuggingFace (this will download the model if not present)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            model_file="mistral-7b-instruct-v0.2.Q4_K_M.gguf",  # Use quantized version for efficiency
            local_files_only=False,
            model_type="mistral",
            max_new_tokens=2048,
            context_length=4096,
            gpu_layers=0  # Use CPU for inference
        )
        
        # Cache the model
        model_cache[model_name] = model
        return None, model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        # Fallback to rule-based methods if model loading fails
        return None, None

def generate_output(text, mode, tokenizer=None, model=None):
    """
    Generate output based on the input text and mode.
    If model is available, use it; otherwise fall back to rule-based methods.
    
    Args:
        text (str): The input policy text
        mode (str): "Simplify" or "Draft"
        tokenizer: Not used with ctransformers
        model: The loaded language model
        
    Returns:
        str: Generated output text
    """
    # If model failed to load, use rule-based approach
    if model is None or model == "model_placeholder":
        if mode == "Simplify":
            return simplify_text_rule_based(text)
        elif mode == "Draft":
            return generate_draft_rule_based(text)
        return text
    
    # Prepare prompt based on mode
    if mode == "Simplify":
        prompt = f"""<s>[INST] You are an expert insurance policy simplifier. Your task is to make complex insurance text extremely easy to understand for anyone. Follow these rules strictly:

1. Break down the text into very short, simple sentences
2. Replace all technical terms with everyday language
3. Use bullet points for better readability
4. Add simple examples where helpful
5. Focus on what matters most to the policyholder
6. Use a friendly, conversational tone
7. Keep each point very brief and clear
8. Highlight important numbers and dates
9. Explain any conditions or limitations in simple terms
10. Use headings to organize different topics

Format your response like this:
# Main Points
• [Very simple summary of the most important points]

# What This Means For You
• [Explain what the policyholder needs to know]

# Important Details
• [Key numbers, dates, or conditions in simple terms]

# Examples
• [Simple examples to illustrate the points]

Please simplify this insurance text to make it extremely easy to understand. Use very simple language and break it down into clear, short points:

{text} [/INST]</s>"""
    else:  # Draft mode
        prompt = f"""<s>[INST] You are an insurance document drafter. Your task is to structure insurance text into a well-organized document.

Please create a well-structured draft document based on this insurance text:

{text} [/INST]</s>"""
    
    try:
        # Generate text with the model
        start_time = time.time()
        result = model(
            prompt,
            max_new_tokens=2048,  # Increased token limit for better summaries
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            stop=["</s>", "[INST]"]
        )
        generation_time = time.time() - start_time
        print(f"Generation completed in {generation_time:.2f} seconds")
        
        # Clean up the result to extract just the assistant's response
        if "[/INST]" in result:
            result = result.split("[/INST]")[1].strip()
        if "</s>" in result:
            result = result.split("</s>")[0].strip()
            
        return result
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        # Fall back to rule-based methods if generation fails
        if mode == "Simplify":
            return simplify_text_rule_based(text)
        elif mode == "Draft":
            return generate_draft_rule_based(text)
        return text

def simplify_text_rule_based(text):
    """
    Performs a rule-based text simplification with comprehensive insurance term explanations.
    
    Args:
        text (str): The input text to simplify
    
    Returns:
        str: Simplified text with explanations
    """
    # Dictionary of insurance terms and their simple explanations
    insurance_terms = {
        # Basic terms
        "policyholder": "person who owns the insurance policy",
        "insured": "person or thing covered by the insurance",
        "premium": "the amount you pay for insurance",
        "deductible": "the amount you pay before insurance starts paying",
        "coverage": "what the insurance will pay for",
        "claim": "request for insurance payment",
        "endorsement": "change or addition to your policy",
        
        # Property terms
        "dwelling": "your home or the building you live in",
        "personal property": "your belongings and stuff you own",
        "other structures": "buildings on your property that aren't your main home",
        "loss of use": "when you can't live in your home and need temporary housing",
        
        # Liability terms
        "liability": "your legal responsibility",
        "negligence": "when you don't take proper care and cause harm",
        "damages": "money paid for harm or loss",
        "defense costs": "legal fees to protect you in court",
        
        # Medical terms
        "medical payments": "money paid for medical treatment",
        "bodily injury": "physical harm to a person",
        "reasonable expenses": "normal and necessary costs",
        
        # Financial terms
        "actual cash value": "what your property is worth now",
        "replacement cost": "what it would cost to buy new",
        "depreciation": "loss in value over time",
        "limit of liability": "maximum amount insurance will pay",
        
        # Condition terms
        "exclusion": "what the insurance doesn't cover",
        "endorsement": "change to your policy",
        "rider": "extra coverage you can add",
        "grace period": "extra time to pay your premium",
        
        # Time-related terms
        "policy period": "how long your insurance lasts",
        "effective date": "when your coverage starts",
        "expiration date": "when your coverage ends",
        "renewal": "continuing your insurance for another period"
    }
    
    # Dictionary of common phrases and their simpler versions
    common_phrases = {
        "in the event of": "if",
        "subject to": "depending on",
        "notwithstanding": "even if",
        "pursuant to": "according to",
        "herein": "in this document",
        "hereinafter": "later in this document",
        "whereas": "while",
        "in accordance with": "following",
        "shall be": "will be",
        "shall not": "will not",
        "in the event that": "if",
        "for the purpose of": "to",
        "with respect to": "about",
        "not less than": "at least",
        "not more than": "no more than",
        "in excess of": "more than",
        "prior to": "before",
        "subsequent to": "after",
        "in lieu of": "instead of",
        "per annum": "per year"
    }
    
    # Split into paragraphs
    paragraphs = text.split('\n\n')
    simplified_paragraphs = []
    
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
            
        # Process each sentence
        sentences = paragraph.split('. ')
        simplified_sentences = []
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # Convert to lowercase for matching
            lower_sentence = sentence.lower()
            
            # Replace common phrases
            for phrase, simple in common_phrases.items():
                if phrase in lower_sentence:
                    sentence = sentence.replace(phrase, simple)
            
            # Replace insurance terms and add explanations
            for term, explanation in insurance_terms.items():
                if term in lower_sentence:
                    # Add explanation in parentheses
                    sentence = sentence.replace(term, f"{term} ({explanation})")
            
            # Break long sentences into shorter ones
            if len(sentence.split()) > 20:
                parts = sentence.split(',')
                if len(parts) > 1:
                    sentence = '. '.join(parts)
            
            simplified_sentences.append(sentence)
        
        # Join sentences back together
        simplified_paragraph = '. '.join(simplified_sentences)
        simplified_paragraphs.append(simplified_paragraph)
    
    # Format the final output
    simplified_text = "\n\n".join(simplified_paragraphs)
    
    # Add explanatory header and footer
    return f"""# SIMPLIFIED EXPLANATION

{simplified_text}

# KEY POINTS TO REMEMBER
• All terms in parentheses are explanations of insurance terms
• This is a simplified version of the original text
• Always check with your insurance provider for exact details
• Keep your original policy document for reference

Note: This simplified version is provided for easier understanding. The original policy terms and conditions still apply."""

def generate_draft_rule_based(text):
    """
    Creates a structured draft document based on input text.
    
    Args:
        text (str): The input text
    
    Returns:
        str: A draft document
    """
    # Create a structured document with sections
    document = [
        "# DRAFT DOCUMENT",
        "",
        "## INTRODUCTION",
        "This document outlines the terms and conditions based on the provided policy text.",
        "",
        "## MAIN PROVISIONS",
        text,
        "",
        "## SUMMARY",
        "The above provisions establish the rights and obligations of all parties involved.",
        "",
        "## NOTES",
        "- This is a draft document generated based on the input text.",
        "- Legal review is recommended before finalization.",
        "- All parties should carefully review terms before agreement.",
        "",
        "Note: This draft was created using a fallback rule-based method."
    ]
    
    return "\n".join(document)

import fitz  # PyMuPDF

def process_file_upload(uploaded_file):
    """
    Process an uploaded file and extract text content.
    
    Args:
        uploaded_file: The file uploaded via Streamlit's file_uploader
        
    Returns:
        str: Text content from the file
    """
    try:
        # Check file type
        file_type = uploaded_file.type
        
        if file_type == "application/pdf":
            pdf_bytes = uploaded_file.read()
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
                return text
        else:
            # Read text content from uploaded file
            content = uploaded_file.getvalue().decode("utf-8")
            return content
    except UnicodeDecodeError:
        raise Exception("The uploaded file could not be read. Please ensure it's a valid text or PDF file.")
    except Exception as e:
        raise Exception(f"Error processing file: {str(e)}")
