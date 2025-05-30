#!/usr/bin/env python3
"""
Test script to demonstrate the new OpenAI model family resolution functionality.

This script shows how users can now specify model families (like "gpt-4o")
and get automatically mapped to the latest stable version, while still
supporting specific version requests.
"""

import os
import sys

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from convorator.client.openai_client import OpenAILLM, MODEL_FAMILY_MAPPING
from convorator.utils.logger import setup_logger


def main():
    """Demonstrate the new model family resolution functionality."""

    print("=" * 80)
    print("OpenAI Client Model Family Resolution Demo")
    print("=" * 80)

    # Set up logger
    logger = setup_logger("test_model_resolution")

    print("\n1. Available Model Families:")
    print("-" * 40)
    families = OpenAILLM.get_available_model_families()
    for family, latest in families.items():
        print(f"  {family:<25} → {latest}")

    print("\n2. Model Resolution Examples:")
    print("-" * 40)

    # Test cases showing model resolution
    test_models = [
        "gpt-4o",  # Should resolve to latest
        "gpt-4.1",  # Should resolve to latest
        "o4-mini",  # Should resolve to latest
        "gpt-4o-2024-05-13",  # Specific version, no resolution
        "gpt-3.5-turbo",  # Should resolve to latest
        "non-existent-model",  # Should remain as-is
    ]

    for model in test_models:
        try:
            # Create client instance (this will resolve the model)
            client = OpenAILLM(
                logger=logger,
                api_key="dummy-key-for-testing",  # Won't be used for resolution demo
                model=model,
            )

            # Get model info
            info = client.get_model_info()

            print(f"  Input: {model:<20}")
            print(f"    → Resolved: {info['resolved_model']}")
            print(f"    → Family resolved: {info['is_family_resolved']}")
            print()

        except Exception as e:
            print(f"  Input: {model:<20} → Error: {e}")

    print("\n3. Backward Compatibility Test:")
    print("-" * 40)

    # Test that existing production code still works
    legacy_models = ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"]

    for model in legacy_models:
        try:
            client = OpenAILLM(logger=logger, api_key="dummy-key-for-testing", model=model)
            info = client.get_model_info()

            print(f"  Legacy model: {model}")
            print(f"    → Still works: ✓")
            print(f"    → Resolved to: {info['resolved_model']}")
            print()

        except Exception as e:
            print(f"  Legacy model: {model} → Error: {e}")

    print("\n4. Provider Capabilities:")
    print("-" * 40)

    # Show new capabilities
    try:
        client = OpenAILLM(logger=logger, api_key="dummy-key-for-testing", model="gpt-4o")

        capabilities = client.get_provider_capabilities()

        print(f"  Supports model families: {capabilities.get('supports_model_families', False)}")
        print(f"  Model family resolution: {client.supports_feature('model_family_resolution')}")
        print(f"  Current model info: {capabilities.get('current_model_info', {})}")

    except Exception as e:
        print(f"  Error getting capabilities: {e}")

    print("\n5. Real-world Usage Examples:")
    print("-" * 40)

    examples = [
        {
            "description": "Use latest GPT-4o (recommended for most applications)",
            "code": 'client = OpenAILLM(logger=logger, model="gpt-4o")',
            "model": "gpt-4o",
        },
        {
            "description": "Use latest reasoning model for complex problems",
            "code": 'client = OpenAILLM(logger=logger, model="o4-mini")',
            "model": "o4-mini",
        },
        {
            "description": "Use specific version for reproducibility",
            "code": 'client = OpenAILLM(logger=logger, model="gpt-4o-2024-05-13")',
            "model": "gpt-4o-2024-05-13",
        },
    ]

    for example in examples:
        try:
            client = OpenAILLM(
                logger=logger, api_key="dummy-key-for-testing", model=example["model"]
            )
            info = client.get_model_info()

            print(f"  Use Case: {example['description']}")
            print(f"    Code: {example['code']}")
            print(f"    Result: {example['model']} → {info['resolved_model']}")
            print()

        except Exception as e:
            print(f"  Example failed: {e}")

    print("=" * 80)
    print("Demo completed! The OpenAI client now supports:")
    print("  ✓ Automatic model family resolution")
    print("  ✓ Latest model version mapping")
    print("  ✓ Backward compatibility with existing code")
    print("  ✓ Specific version support when needed")
    print("=" * 80)


if __name__ == "__main__":
    main()
