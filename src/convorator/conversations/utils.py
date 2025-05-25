import inspect
import json
import re
from typing import Dict, Optional, Union, get_args, get_origin

import jsonschema

from jsonschema.exceptions import ValidationError, SchemaError


# Import exceptions from the central location
from convorator.conversations.types import LoggerProtocol
from convorator.exceptions import LLMResponseError, SchemaValidationError

Schema = Dict[str, object]


def validate_json(logger: LoggerProtocol, data: Dict[str, object], schema: Schema) -> None:
    """
    Validate JSON data against a given schema.

    Args:
        logger: Logger instance
        data (dict): JSON data to validate
        schema (dict): JSON schema for validation

    Raises:
        SchemaValidationError: If validation fails.
        jsonschema.exceptions.SchemaError: If the schema itself is invalid.
    """
    try:
        # Validate the data against the schema
        jsonschema.validate(instance=data, schema=schema)
        logger.debug("JSON validation successful.")
    except ValidationError as validation_err:
        # Construct a more informative error message
        error_msg = f"JSON Validation Error: {validation_err.message} at path '{'/'.join(map(str, validation_err.path))}'. Schema path: '{'/'.join(map(str, validation_err.schema_path))}'. Instance snippet: {str(validation_err.instance)[:100]}..."
        logger.error(error_msg)
        # Wrap the original validation error in our custom exception
        raise SchemaValidationError(error_msg, schema=schema, instance=data) from validation_err
    except SchemaError as schema_err:
        # Schema errors are usually programmer errors, re-raise directly but log
        logger.error(f"Invalid JSON Schema provided: {schema_err}", exc_info=True)
        raise  # Re-raise the original SchemaError


def parse_json_response(
    logger: LoggerProtocol,
    response: str,
    context: str,
    schema: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """
    Robustly parse JSON from LLM response and validate against schema if provided.

    Args:
        logger: Logger instance
        response: Raw LLM response string
        context: Context string for error messages (e.g., "parsing LLM A response")
        schema: Optional JSON schema for validation

    Returns:
        Dict[str, Any]: Parsed and validated JSON dictionary

    Raises:
        LLMResponseError: If JSON parsing fails or no JSON is found in the response.
        SchemaValidationError: If JSON validation against the schema fails.
        jsonschema.exceptions.SchemaError: If the provided schema itself is invalid.
    """
    logger.debug(f"Attempting to parse JSON from response in context: {context}")
    result_json = None
    try:
        # Try to find JSON within ```json ... ``` blocks first
        # Use findall to get all non-overlapping matches
        json_matches = re.findall(r"```json\s*([\s\S]*?)\s*```", response, re.IGNORECASE)
        if json_matches:
            # Use the last match found
            json_str = json_matches[-1].strip()
            logger.debug(f"Found JSON in last ```json block: {json_str[:100]}...")
            result_json = json.loads(json_str)
        else:
            # If no block found, try to find the first valid JSON object/array
            # This is less robust but can catch JSON not in code blocks
            json_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", response)
            if json_match:
                # Be careful with greedy matching, try parsing the first potential match
                potential_json = json_match.group(0)
                try:
                    result_json = json.loads(potential_json)
                    logger.debug(f"Found potential JSON object/array: {potential_json[:100]}...")
                except json.JSONDecodeError:
                    logger.warning(
                        f"Found potential JSON-like structure in {context} but failed to parse."
                    )
                    # Continue to raise error below if result_json is still None

        if result_json is None:
            logger.error(f"No valid JSON found in {context} response.")
            raise LLMResponseError(f"No valid JSON found in {context} response.")

        # Validate against schema if provided
        if schema:
            logger.debug(f"Validating parsed JSON against schema in context: {context}")
            validate_json(
                logger, result_json, schema
            )  # This will raise SchemaValidationError on failure

        logger.debug(f"Successfully parsed and validated JSON in context: {context}")
        return result_json

    except json.JSONDecodeError as e:
        error_msg = (
            f"JSON parsing error in {context} response: {e}. Response snippet: {response[:200]}"
        )
        logger.error(error_msg)
        raise LLMResponseError(error_msg, original_exception=e)
    except SchemaValidationError:  # Re-raise schema validation errors
        raise
    except SchemaError:  # Re-raise schema definition errors
        raise
    except Exception as e:  # Catch other unexpected errors during parsing
        # Avoid wrapping errors we already handle specifically
        if isinstance(e, (LLMResponseError, SchemaValidationError)):
            raise
        error_msg = f"Unexpected error parsing JSON in {context}: {e}"
        logger.error(error_msg, exc_info=True)
        raise LLMResponseError(error_msg, original_exception=e)


def is_optional_type(param: inspect.Parameter) -> bool:
    """
    Check if a parameter has an Optional[] type annotation.

    Works with:
    - Optional[X]
    - Union[X, None]
    - Union[None, X]
    - X | None (Python 3.10+)
    """
    # No annotation means it's not explicitly optional
    if param.annotation is param.empty:
        return False

    # Check for Python 3.10+ union syntax (X | None)
    if get_origin(param.annotation) is Union:
        # Handle Union[X, None] or Union[None, X]
        args = get_args(param.annotation)
        return type(None) in args

    # Check for direct Optional[X]
    if get_origin(param.annotation) is Optional:
        return True

    return False
