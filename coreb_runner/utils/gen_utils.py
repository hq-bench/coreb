import json
import re
from easyllm_kit.utils import get_logger, extract_json_from_text
from typing import List, Dict
import json_repair

logger = get_logger('code_emb_bench')


def _prepare_litellm_message(prompt: str) -> List[Dict]:
    """Helper function to prepare message for LiteLLM-based models."""
    message = [{"type": "text", "text": prompt}]
    return message


def generate_response_from_llm(
        model,
        input_prompt: str
) -> str:
    """
    Generate responses from various LLM models with optional image input and OCR processing.

    Args:
        model: The language model instance to use for generation
        input_prompt (str): The text prompt to send to the model

    Returns:
        str: The generated response from the model

    Raises:
        ValueError: If the model name is not supported
        NotImplementedError: If the model type is not implemented
    """
    if not hasattr(model, 'model_name'):
        raise ValueError("Model must have 'model_name' attribute")

    if model.model_name in ['qwen', 'qwen_vl']:
        response = model.generate(input_prompt)
        try:
            return json_repair.loads(response)["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Failed to parse Qwen model response: {e}")
    elif model.model_name == 'gemini' and not model.model_config.use_litellm_api:
        return model.generate(input_prompt)
    else:
        message = _prepare_litellm_message(input_prompt)
        return model.generate(message)


def safe_parse_response(response_text, reasoning_attached=False):
    """
    Parse the response string as JSON or extracts data using regex if JSON parsing fails.
    Args:
        response_text: The text response from the model
        reasoning_attached: whether the reasoning is attached in the response text
    Returns:
        Dictionary mapping question IDs to their answers and explanations
    """
    # Initialize response dictionary
    response_dict = {}

    # Extract reasoning if attached
    reasoning = ""
    if reasoning_attached and '<reason>' in response_text and '</reason>' in response_text:
        try:
            reasoning = response_text.split('<reason>')[-1].split('</reason>')[0]
            response_text = response_text.split('<reason>')[0]
            response_dict['reasoning'] = reasoning
        except Exception as e:
            logger.warning(f"Error extracting reasoning: {e}")
            # Continue with parsing even if reasoning extraction fails

    # Try to parse as JSON
    try:
        parsed_json = json_repair.loads(response_text)
        response_dict.update(parsed_json)
    except json.JSONDecodeError:
        try:
            extracted_json = extract_json_from_text(response_text)
            response_dict.update(extracted_json)
        except Exception as e:
            logger.warning(f"Error extracting JSON: {e}")
            response_dict['result'] = 'error parsing'

    # Ensure reasoning is preserved in the final output
    if reasoning and 'reasoning' not in response_dict:
        response_dict['reasoning'] = reasoning

    return response_dict
