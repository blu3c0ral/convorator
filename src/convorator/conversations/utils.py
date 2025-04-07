import inspect
import json
import logging
import re
from typing import Any, Callable, Dict, Optional, TypeVar, Union, get_args, get_origin

import jsonschema

from convorator.conversations.conversation_setup import LLMResponseError, SchemaValidationError
from convorator.exceptions import MissingVariableError


def validate_json(logger: logging.Logger, data: Dict, schema: Dict) -> str:
    """
    Validate JSON data against a given schema.

    Args:
        logger: Logger instance
        data (dict): JSON data to validate
        schema (dict): JSON schema for validation

    Raises:
        jsonschema.exceptions.SchemaError: If the schema itself is invalid
    """
    try:
        # Validate the data against the schema
        jsonschema.validate(instance=data, schema=schema)
        logger.debug("JSON validation successful.")
    except jsonschema.exceptions.ValidationError as validation_err:
        error_msg = f"JSON Validation Error: {validation_err.message} in instance {validation_err.instance}. Schema path: {list(validation_err.schema_path)}"
        logger.error(error_msg)
        # Wrap the original validation error
        raise SchemaValidationError(error_msg, validation_error=validation_err)
    except jsonschema.exceptions.SchemaError as schema_err:
        # Schema errors are usually programmer errors, re-raise directly
        logger.error(f"Invalid JSON Schema provided: {schema_err}", exc_info=True)
        raise


def parse_json_response(
    logger: logging.Logger,
    response: str,
    context: str,
    schema: Optional[Dict] = None,
) -> Dict:
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
        json_match = re.search(r"```json\s*([\s\S]*?)\s*```", response, re.IGNORECASE)
        if json_match:
            json_str = json_match.group(1).strip()
            logger.debug(f"Found JSON in ```json block: {json_str[:100]}...")
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
    except jsonschema.exceptions.SchemaError:  # Re-raise schema definition errors
        raise
    except Exception as e:  # Catch unexpected errors during parsing
        error_msg = f"Unexpected error parsing JSON in {context}: {e}"
        logger.error(error_msg, exc_info=True)
        raise LLMResponseError(error_msg, original_exception=e)


T = TypeVar("T")


def execute_with_locals(func: Callable[..., T]) -> T:
    """
    Call a function with arguments extracted from a dictionary of local variables.

    Intelligently handles optional parameters:
    - Parameters with default values
    - Parameters with Optional[] type hints
    - Parameters with default value None

    Args:
        func: The function to call

    Returns:
        The result of calling the function with matched parameters
    """
    # Get the current stack frame
    stack = inspect.stack()
    # stack[0] is the current function
    # stack[1] is the immediate caller
    caller = stack[1]
    # Get caller local variables
    local_vars = caller.frame.f_locals

    # Get function signature
    sig = inspect.signature(func)

    # Build arguments dict by matching parameter names with local variables
    args = {}
    for param_name, param in sig.parameters.items():
        if param_name in local_vars:
            # Local variable exists, use it regardless of optionality
            args[param_name] = local_vars[param_name]
        elif param.default is not param.empty:
            # Parameter has default value, so it's optional - skip it
            continue
        elif is_optional_type(param):
            # Parameter has Optional[] type hint but no default - it's still optional
            continue
        else:
            # Required parameter is missing
            raise MissingVariableError(
                f"Missing required variable '{param_name}' in locals() for function '{func.__name__}' called from {caller.function} in {caller.filename} at line {caller.lineno}."
                f" Locals: {local_vars}"
            )

    # Call the function with the collected arguments
    return func(**args)


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
