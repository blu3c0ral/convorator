from setuptools import setup, find_packages

setup(
    name="llm-conversation-framework",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.25.0",
        "jsonschema>=4.0.0",
    ],
    extras_require={
        "dev": [
            "black",
            "flake8",
            "isort",
            "mypy",
        ],
        "test": [
            "pytest",
            "pytest-cov",
            "coverage",
        ],
    },
)
