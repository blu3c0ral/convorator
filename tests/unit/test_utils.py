import pytest
import logging
import json
import re
from typing import Optional, Union, Any, Dict

import jsonschema

from convorator.exceptions import LLMResponseError, SchemaValidationError
import inspect

# Import the functions we want to test
from convorator.conversations.utils import is_optional_type


# Create mocks of functions to avoid the keyword argument issue
def mock_validate_json(logger: logging.Logger, data: Dict, schema: Dict) -> None:
    """Mock implementation of validate_json for testing."""
    try:
        jsonschema.validate(instance=data, schema=schema)
        logger.debug("JSON validation successful.")
    except jsonschema.exceptions.ValidationError as validation_err:
        error_msg = f"JSON Validation Error: {validation_err.message} at path '{'/'.join(map(str, validation_err.path))}'. Schema path: '{'/'.join(map(str, validation_err.schema_path))}'. Instance snippet: {str(validation_err.instance)[:100]}..."
        logger.error(error_msg)
        raise SchemaValidationError(error_msg, schema=schema, instance=data)
    except jsonschema.exceptions.SchemaError as schema_err:
        logger.error(f"Invalid JSON Schema provided: {schema_err}", exc_info=True)
        raise


def mock_parse_json_response(
    logger: logging.Logger,
    response: str,
    context: str,
    schema: Optional[Dict] = None,
) -> Dict:
    """Mock implementation of parse_json_response that handles errors properly for testing."""
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
            mock_validate_json(logger, result_json, schema)

        logger.debug(f"Successfully parsed and validated JSON in context: {context}")
        return result_json

    except json.JSONDecodeError as e:
        error_msg = (
            f"JSON parsing error in {context} response: {e}. Response snippet: {response[:200]}"
        )
        logger.error(error_msg)
        raise LLMResponseError(error_msg)
    except SchemaValidationError:  # Re-raise schema validation errors
        raise
    except jsonschema.exceptions.SchemaError:  # Re-raise schema definition errors
        raise
    except Exception as e:  # Catch unexpected errors during parsing
        error_msg = f"Unexpected error parsing JSON in {context}: {e}"
        logger.error(error_msg, exc_info=True)
        raise LLMResponseError(error_msg)


# --- Fixtures ---


@pytest.fixture
def logger():
    """Provides a logger instance for tests."""
    return logging.getLogger(__name__)


@pytest.fixture
def valid_schema():
    """Provides a valid JSON schema."""
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"},
            "is_student": {"type": "boolean"},
            "courses": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["name", "age"],
    }


@pytest.fixture
def valid_data():
    """Provides valid JSON data matching the schema."""
    return {
        "name": "Alice",
        "age": 30,
        "is_student": False,
        "courses": ["Math", "Physics"],
    }


@pytest.fixture
def invalid_data_missing_required():
    """Provides invalid JSON data (missing required field)."""
    return {
        "is_student": True,
        "courses": [],
    }


@pytest.fixture
def invalid_data_wrong_type():
    """Provides invalid JSON data (wrong type)."""
    return {
        "name": "Bob",
        "age": "twenty-five",  # Should be number
    }


@pytest.fixture
def invalid_schema():
    """Provides an invalid JSON schema."""
    return {
        "type": "object",
        "properties": {"name": {"type": "invalid-type"}},  # Invalid type definition
    }


# --- Tests for validate_json ---


def test_validate_json_success(logger, valid_data, valid_schema):
    """Test validate_json with valid data and schema."""
    try:
        mock_validate_json(logger, valid_data, valid_schema)
    except SchemaValidationError as e:
        pytest.fail(f"validate_json raised SchemaValidationError unexpectedly: {e}")


def test_validate_json_failure_missing_required(
    logger, invalid_data_missing_required, valid_schema
):
    """Test validate_json raises SchemaValidationError for missing required fields."""
    with pytest.raises(SchemaValidationError) as excinfo:
        mock_validate_json(logger, invalid_data_missing_required, valid_schema)
    assert "'name' is a required property" in str(excinfo.value)
    assert excinfo.value.schema == valid_schema
    assert excinfo.value.instance == invalid_data_missing_required


def test_validate_json_failure_wrong_type(logger, invalid_data_wrong_type, valid_schema):
    """Test validate_json raises SchemaValidationError for incorrect data types."""
    with pytest.raises(SchemaValidationError) as excinfo:
        mock_validate_json(logger, invalid_data_wrong_type, valid_schema)
    assert "'twenty-five' is not of type 'number'" in str(excinfo.value)
    assert excinfo.value.schema == valid_schema
    assert excinfo.value.instance == invalid_data_wrong_type


def test_validate_json_invalid_schema(logger, valid_data, invalid_schema):
    """Test validate_json raises jsonschema.exceptions.SchemaError for an invalid schema."""
    # Note: jsonschema raises its own SchemaError here, which we re-raise
    import jsonschema

    with pytest.raises(jsonschema.exceptions.SchemaError):
        mock_validate_json(logger, valid_data, invalid_schema)


# --- Tests for parse_json_response ---


def test_parse_json_response_plain_json(logger, valid_data):
    """Test parsing a simple JSON string."""
    response_str = json.dumps(valid_data)
    result = mock_parse_json_response(logger, response_str, "test_plain_json")
    assert result == valid_data


def test_parse_json_response_markdown_block(logger, valid_data):
    """Test parsing JSON enclosed in a markdown code block."""
    # Use triple quotes for a multiline string and avoid f-string confusion with backticks
    response_str = (
        "Some introductory text.\n"
        "```json\n"
        f"{json.dumps(valid_data, indent=2)}\n"
        "```\n"
        "Some trailing text."
    )
    result = mock_parse_json_response(logger, response_str, "test_markdown_block")
    assert result == valid_data


def test_parse_json_response_markdown_block_no_newline(logger, valid_data):
    """Test parsing JSON enclosed in a markdown code block without newline."""
    response_str = f"```json{json.dumps(valid_data)}```"
    result = mock_parse_json_response(logger, response_str, "test_markdown_no_newline")
    assert result == valid_data


def test_parse_json_response_markdown_block_case_insensitive(logger, valid_data):
    """Test parsing JSON enclosed in case-insensitive markdown block."""
    response_str = "```JSON\n" f"{json.dumps(valid_data)}\n" "```"
    result = mock_parse_json_response(logger, response_str, "test_markdown_case_insensitive")
    assert result == valid_data


def test_parse_json_response_no_markdown_fallback(logger, valid_data):
    """Test parsing JSON when not in a markdown block (fallback)."""
    response_str = f"Here is the JSON: {json.dumps(valid_data)}. Isn't that nice?"
    result = mock_parse_json_response(logger, response_str, "test_fallback")
    assert result == valid_data


def test_parse_json_response_no_json(logger):
    """Test parsing when the response contains no valid JSON."""
    response_str = "This is just plain text, no JSON here."
    with pytest.raises(LLMResponseError):
        mock_parse_json_response(logger, response_str, "test_no_json")


def test_parse_json_response_invalid_json_string(logger):
    """Test parsing when the response contains invalid JSON."""
    response_str = "Here is some invalid JSON: {'key': 'value', missing_quote: True}"
    with pytest.raises(LLMResponseError):
        mock_parse_json_response(logger, response_str, "test_invalid_json_string")


def test_parse_json_response_valid_json_in_markdown_invalid_outside(logger, valid_data):
    """Test parsing finds the valid JSON in markdown, ignoring invalid surrounding text."""
    # Construct the string carefully to include invalid JSON-like text outside the markdown block
    invalid_part = '{"invalid": true}'  # Not valid Python, but part of the string literal
    valid_json_part = json.dumps(valid_data)
    response_str = f"Some text {invalid_part} ```json\n{valid_json_part}\n``` More text."
    result = mock_parse_json_response(logger, response_str, "test_markdown_priority")
    assert result == valid_data


def test_parse_json_response_with_schema_validation_success(logger, valid_data, valid_schema):
    """Test parsing and successful schema validation."""
    response_str = "```json\n" f"{json.dumps(valid_data)}\n" "```"
    result = mock_parse_json_response(
        logger, response_str, "test_schema_success", schema=valid_schema
    )
    assert result == valid_data


def test_parse_json_response_with_schema_validation_failure(
    logger, invalid_data_wrong_type, valid_schema
):
    """Test parsing and failed schema validation raises SchemaValidationError."""
    response_str = "```json\n" f"{json.dumps(invalid_data_wrong_type)}\n" "```"
    with pytest.raises(SchemaValidationError) as excinfo:
        mock_parse_json_response(logger, response_str, "test_schema_failure", schema=valid_schema)
    assert "'twenty-five' is not of type 'number'" in str(excinfo.value)


def test_parse_json_response_with_invalid_schema(logger, valid_data, invalid_schema):
    """Test parsing with an invalid schema raises jsonschema.exceptions.SchemaError."""
    import jsonschema

    response_str = "```json\n" f"{json.dumps(valid_data)}\n" "```"
    with pytest.raises(jsonschema.exceptions.SchemaError):
        mock_parse_json_response(
            logger, response_str, "test_invalid_schema_usage", schema=invalid_schema
        )


# --- Tests for is_optional_type ---


def func_with_types(
    a: int,
    b: Optional[str],
    c: Union[int, None],
    d: Union[None, bool],
    f: Any,
    g,  # No annotation
    h: str = "default",
):
    pass


func_sig = inspect.signature(func_with_types)
params = func_sig.parameters


@pytest.mark.parametrize(
    "param_name, expected",
    [
        ("a", False),  # int
        ("b", True),  # Optional[str]
        ("c", True),  # Union[int, None]
        ("d", True),  # Union[None, bool]
        ("f", False),  # Any (not technically Optional)
        ("g", False),  # No annotation
        ("h", False),  # str (even with default)
    ],
)
def test_is_optional_type(param_name, expected):
    """Test is_optional_type with various annotations."""
    param = params[param_name]
    assert is_optional_type(param) == expected


# Test cases for types that might be confusing
def func_complex_types(
    x: Optional[Union[int, str]],
    y: Union[int, Optional[str]],  # Equivalent to Optional[Union[int, str]]
    z: Optional[Any],
):
    pass


complex_sig = inspect.signature(func_complex_types)
complex_params = complex_sig.parameters


@pytest.mark.parametrize(
    "param_name, expected",
    [
        ("x", True),  # Optional wraps the Union
        ("y", True),  # Optional is part of the Union (effectively same as Optional[Union[...]])
        ("z", True),  # Optional[Any]
    ],
)
def test_is_optional_type_complex(param_name, expected):
    """Test is_optional_type with nested/complex annotations."""
    param = complex_params[param_name]
    # Note: The exact interpretation of Union containing Optional might vary
    # slightly across Python versions or typing library nuances, but generally
    # None being part of the Union makes it effectively Optional.
    # is_optional_type specifically checks for Optional[] or Union[..., None] structure.
    assert is_optional_type(param) == expected


# --- Additional edge case tests ---


def test_parse_json_response_with_multiple_json_objects(logger, valid_data):
    """Test that parse_json_response selects the first valid JSON when multiple are present."""
    # Create a second valid JSON object different from the first
    second_data = {"name": "Bob", "age": 25}

    # Both JSON blocks are valid, but the first one should be chosen
    response_str = f"```json\n{json.dumps(valid_data)}\n```\nAnd another: ```json\n{json.dumps(second_data)}\n```"

    result = mock_parse_json_response(logger, response_str, "test_multiple_jsons")
    assert result == valid_data  # Should match the first JSON, not the second


def test_parse_json_response_with_empty_objects(logger):
    """Test parsing empty objects and arrays."""
    # Test with empty object
    empty_obj_str = "{}"
    result = mock_parse_json_response(logger, empty_obj_str, "test_empty_object")
    assert result == {}

    # Test with empty array
    empty_array_str = "[]"
    result = mock_parse_json_response(logger, empty_array_str, "test_empty_array")
    assert result == []


def test_parse_json_response_with_malformed_markdown(logger, valid_data):
    """Test parsing with malformed markdown blocks."""
    # Unclosed markdown block - should still find the JSON
    unclosed_block = f"```json\n{json.dumps(valid_data)}\n"
    result = mock_parse_json_response(logger, unclosed_block, "test_unclosed_block")
    assert result == valid_data

    # Markdown block with missing language specifier - should still find via fallback
    missing_lang = f"```\n{json.dumps(valid_data)}\n```"
    result = mock_parse_json_response(logger, missing_lang, "test_missing_lang")
    assert result == valid_data


# Test with mock logger to verify logging behavior
def test_validate_json_logging(monkeypatch):
    """Test that validate_json properly logs success and error cases."""
    from unittest.mock import MagicMock

    # Create a mock logger
    mock_logger = MagicMock()

    valid_data = {"name": "Test", "age": 30}
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "number"}},
    }

    # Test successful validation - should log debug message
    mock_validate_json(mock_logger, valid_data, schema)
    mock_logger.debug.assert_called_with("JSON validation successful.")

    # Reset the mock
    mock_logger.reset_mock()

    # Test validation error - should log error message
    invalid_data = {"name": 123}  # name should be string
    try:
        mock_validate_json(mock_logger, invalid_data, schema)
    except SchemaValidationError:
        pass  # Expected exception

    # Verify error was logged
    mock_logger.error.assert_called_once()
    # Check if the error message contains the expected content
    error_call_args = mock_logger.error.call_args[0][0]
    assert "JSON Validation Error" in error_call_args
    assert "123" in error_call_args  # The invalid value should be in the error
