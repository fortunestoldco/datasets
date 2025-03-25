#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hugging Face and GitHub Organization Documentation Downloader

This script downloads documentation, examples, and cookbook folders from all repositories
of a given organization, converts Python and Jupyter notebook files to Markdown format,
and organizes them into train and test sets.

Enhanced with:
- Real-time console logging with progress bars
- Multiprocessing for faster file processing 
- Text-based user interface (TUI) for managing credentials and repositories
- Schema standardization to fix dataset generation errors
- SFT-ready dataset preparation
- Code generation dataset with SDK-specific features
- Processing of existing downloaded data
"""

import os
import shutil
import random
import datetime
import tempfile
import csv
import re
import json
import traceback
import requests
import uuid
import warnings
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Any, Optional, Set, Union
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("doc_downloader.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Try importing required libraries, install if missing
REQUIRED_LIBRARIES = [
    "huggingface_hub", "datasets", "nbformat", "nbconvert", 
    "gitpython", "tqdm", "prompt_toolkit"
]

try:
    from huggingface_hub import HfApi, Repository, hf_hub_download, snapshot_download
    from datasets import Dataset, DatasetDict
    import nbformat
    from nbconvert import MarkdownExporter
    import git
    from tqdm import tqdm
    from tqdm.contrib.concurrent import process_map
    from prompt_toolkit import prompt
    from prompt_toolkit.shortcuts import checkboxlist_dialog, radiolist_dialog, input_dialog, message_dialog
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.styles import Style
except ImportError as e:
    missing_lib = str(e).split("'")[1]
    logger.error(f"Required library {missing_lib} not found. Installing required libraries...")
    
    import subprocess
    for lib in REQUIRED_LIBRARIES:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
        except subprocess.CalledProcessError:
            logger.error(f"Failed to install {lib}. Please install it manually: pip install {lib}")
            sys.exit(1)
    
    # Try imports again after installation
    from huggingface_hub import HfApi, Repository, hf_hub_download, snapshot_download
    from datasets import Dataset, DatasetDict
    import nbformat
    from nbconvert import MarkdownExporter
    import git
    from tqdm import tqdm
    from tqdm.contrib.concurrent import process_map
    from prompt_toolkit import prompt
    from prompt_toolkit.shortcuts import checkboxlist_dialog, radiolist_dialog, input_dialog, message_dialog
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.styles import Style

# Configuration file path
CONFIG_FILE = os.path.expanduser("~/.doc_downloader_config.json")

# Define TUI styles
STYLE = Style.from_dict({
    'dialog': 'bg:#007733 #ffffff',
    'dialog.body': 'bg:#007733 #ffffff',
    'dialog.frame.label': 'bg:#003300 #ffffff',
    'dialog.body label': '#ffffff',
    'dialog shadow': 'bg:#003300',
})

# Utility functions for document filtering and schema standardization
def is_documentation_file(file_path: str, content: str = None) -> bool:
    """
    Check if a file is likely a documentation file.
    
    Args:
        file_path (str): Path to the file
        content (str): File content (optional, will read from file if not provided)
        
    Returns:
        bool: True if the file appears to be documentation
    """
    # Skip JSON files with API specifications unless they're properly processed
    if file_path.endswith('.json'):
        if content is None:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except Exception:
                return False
        
        # Check if this looks like an OpenAPI spec - these are handled specially elsewhere
        openapi_indicators = ['swagger', 'openapi', 'paths', 'components', 'servers']
        try:
            data = json.loads(content)
            if isinstance(data, dict) and any(key in data for key in openapi_indicators):
                # We'll consider API specs as documentation if they've been converted to our format
                return file_path.endswith('_api_spec.md')
        except Exception:
            pass  # Not valid JSON or other issue, continue checks
    
    # Get extension
    ext = os.path.splitext(file_path)[1].lower()
    
    # Markdown, RST, and TXT files are likely documentation
    if ext in ['.md', '.rst', '.txt']:
        return True
    
    # For Python and other files, check if they've been converted to markdown already
    if content is None:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception:
            return False
    
    # Check for markdown-like content
    md_indicators = ['# ', '## ', '```', '**', '__']
    if any(indicator in content for indicator in md_indicators):
        return True
    
    return False

def standardize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure all metadata entries have the same schema.
    
    Args:
        metadata (Dict[str, Any]): Metadata dictionary
        
    Returns:
        Dict[str, Any]: Standardized metadata
    """
    # Define the standard schema
    standard_fields = {
        'file_name': '',
        'file_path': '',
        'repo_name': '',
        'category': '',
        'title': '',
        'description': '',
        'file_type': '',
        'file_size': 0,
        'creation_date': '',
        'modification_date': '',
        'content': '',
        'split': '',
        'is_api_spec': False,
        'is_sdk_file': False
    }
    
    # Ensure all standard fields exist
    result = {**standard_fields, **{k: v for k, v in metadata.items() if k in standard_fields}}
    
    return result

# Function to create or load configuration
def load_or_create_config() -> Dict[str, Any]:
    """
    Load configuration from file or create a new one if it doesn't exist.
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    default_config = {
        'github_token': '',
        'huggingface_token': '',
        'repositories': [],
        'output_directory': './downloaded_docs',
        'test_ratio': 0.2
    }
    
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            
            # Ensure all keys exist in the loaded config
            for key in default_config:
                if key not in config:
                    config[key] = default_config[key]
            
            return config
        except Exception as e:
            logger.error(f"Error loading configuration file: {str(e)}")
            return default_config
    else:
        return default_config

# Function to save configuration
def save_config(config: Dict[str, Any]) -> None:
    """
    Save configuration to file.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
    """
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Secure the file permissions (read/write only for the user)
        os.chmod(CONFIG_FILE, 0o600)
    except Exception as e:
        logger.error(f"Error saving configuration file: {str(e)}")

# TUI Functions
def manage_credentials(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Manage GitHub and Hugging Face credentials through TUI.
    
    Args:
        config (Dict[str, Any]): Current configuration
        
    Returns:
        Dict[str, Any]: Updated configuration
    """
    while True:
        # Display current tokens (masked)
        github_token_display = "********" if config['github_token'] else "[Not set]"
        huggingface_token_display = "********" if config['huggingface_token'] else "[Not set]"
        
        choice = radiolist_dialog(
            title="Manage Credentials",
            text="Select an option:",
            values=[
                ("github", f"GitHub Token ({github_token_display})"),
                ("huggingface", f"Hugging Face Token ({huggingface_token_display})"),
                ("back", "Go Back")
            ],
            style=STYLE
        ).run()
        
        if choice == "github":
            token = prompt("Enter your GitHub token (leave empty to keep existing): ", 
                           is_password=True)
            if token:
                config['github_token'] = token
                save_config(config)
                tqdm.write("GitHub token updated successfully.")
        
        elif choice == "huggingface":
            token = prompt("Enter your Hugging Face token (leave empty to keep existing): ", 
                           is_password=True)
            if token:
                config['huggingface_token'] = token
                save_config(config)
                tqdm.write("Hugging Face token updated successfully.")
        
        elif choice == "back" or choice is None:
            break
    
    return config

def manage_repositories(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Manage list of repositories through TUI.
    
    Args:
        config (Dict[str, Any]): Current configuration
        
    Returns:
        Dict[str, Any]: Updated configuration
    """
    while True:
        choice = radiolist_dialog(
            title="Manage Repositories",
            text="Select an option:",
            values=[
                ("add", "Add Repository"),
                ("remove", "Remove Repository"),
                ("list", "List Repositories"),
                ("back", "Go Back")
            ],
            style=STYLE
        ).run()
        
        if choice == "add":
            repo_type = radiolist_dialog(
                title="Repository Type",
                text="Select repository type:",
                values=[
                    ("org", "Organization (All Repositories)"),
                    ("repo", "Single Repository")
                ],
                style=STYLE
            ).run()
            
            if repo_type == "org":
                org_name = input_dialog(
                    title="Add Organization",
                    text="Enter organization name:",
                    style=STYLE
                ).run()
                
                if org_name:
                    source = radiolist_dialog(
                        title="Repository Source",
                        text="Select source:",
                        values=[
                            ("github", "GitHub"),
                            ("huggingface", "Hugging Face Hub")
                        ],
                        style=STYLE
                    ).run()
                    
                    if source:
                        new_repo = {
                            "type": "organization",
                            "name": org_name,
                            "source": source
                        }
                        
                        # Check if already exists
                        if new_repo not in config['repositories']:
                            config['repositories'].append(new_repo)
                            save_config(config)
                            tqdm.write(f"Added organization: {org_name} ({source})")
                        else:
                            tqdm.write(f"Organization {org_name} ({source}) already exists.")
            
            elif repo_type == "repo":
                repo_url = input_dialog(
                    title="Add Repository",
                    text="Enter repository URL:",
                    style=STYLE
                ).run()
                
                if repo_url:
                    # Validate URL format
                    if "github.com" in repo_url:
                        source = "github"
                    elif "huggingface.co" in repo_url:
                        source = "huggingface"
                    else:
                        source = radiolist_dialog(
                            title="Repository Source",
                            text="Cannot determine source from URL. Select source:",
                            values=[
                                ("github", "GitHub"),
                                ("huggingface", "Hugging Face Hub")
                            ],
                            style=STYLE
                        ).run()
                    
                    if source:
                        # Extract repo name from URL
                        if repo_url.endswith("/"):
                            repo_url = repo_url[:-1]
                        
                        repo_name = repo_url.split("/")[-1]
                        org_name = repo_url.split("/")[-2]
                        
                        new_repo = {
                            "type": "repository",
                            "name": f"{org_name}/{repo_name}",
                            "source": source,
                            "url": repo_url
                        }
                        
                        # Check if already exists
                        if not any(r.get('url') == repo_url for r in config['repositories']):
                            config['repositories'].append(new_repo)
                            save_config(config)
                            tqdm.write(f"Added repository: {repo_name}")
                        else:
                            tqdm.write(f"Repository {repo_name} already exists.")
        
        elif choice == "remove":
            if not config['repositories']:
                tqdm.write("No repositories to remove.")
                continue
                
            # Create a list of repositories to choose from
            repo_list = []
            for i, repo in enumerate(config['repositories']):
                if repo['type'] == 'organization':
                    label = f"{repo['name']} (Organization, {repo['source']})"
                else:
                    label = f"{repo['name']} (Repository, {repo['source']})"
                repo_list.append((i, label))
            
            # Let user select repositories to remove
            selected = checkboxlist_dialog(
                title="Remove Repositories",
                text="Select repositories to remove:",
                values=repo_list,
                style=STYLE
            ).run()
            
            if selected:
                # Remove selected repositories (in reverse order to avoid index shifting)
                for idx in sorted(selected, reverse=True):
                    removed = config['repositories'].pop(idx)
                    if removed['type'] == 'organization':
                        tqdm.write(f"Removed organization: {removed['name']}")
                    else:
                        tqdm.write(f"Removed repository: {removed['name']}")
                
                save_config(config)
        
        elif choice == "list":
            if not config['repositories']:
                tqdm.write("No repositories configured.")
            else:
                tqdm.write("\nConfigured Repositories:")
                for i, repo in enumerate(config['repositories'], 1):
                    if repo['type'] == 'organization':
                        tqdm.write(f"{i}. {repo['name']} (Organization, {repo['source']})")
                    else:
                        tqdm.write(f"{i}. {repo['name']} (Repository, {repo['source']})")
                tqdm.write("")
        
        elif choice == "back" or choice is None:
            break
    
    return config

def configure_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Configure general settings through TUI.
    
    Args:
        config (Dict[str, Any]): Current configuration
        
    Returns:
        Dict[str, Any]: Updated configuration
    """
    while True:
        choice = radiolist_dialog(
            title="Configure Settings",
            text="Select a setting to configure:",
            values=[
                ("output", f"Output Directory ({config['output_directory']})"),
                ("test_ratio", f"Test Split Ratio ({config['test_ratio']})"),
                ("back", "Go Back")
            ],
            style=STYLE
        ).run()
        
        if choice == "output":
            output_dir = input_dialog(
                title="Output Directory",
                text="Enter the output directory path:",
                default=config['output_directory'],
                style=STYLE
            ).run()
            
            if output_dir:
                config['output_directory'] = output_dir
                save_config(config)
                tqdm.write(f"Output directory updated to: {output_dir}")
        
        elif choice == "test_ratio":
            test_ratio_str = input_dialog(
                title="Test Split Ratio",
                text="Enter the test split ratio (between 0 and 1):",
                default=str(config['test_ratio']),
                style=STYLE
            ).run()
            
            if test_ratio_str:
                try:
                    test_ratio = float(test_ratio_str)
                    if 0 <= test_ratio <= 1:
                        config['test_ratio'] = test_ratio
                        save_config(config)
                        tqdm.write(f"Test split ratio updated to: {test_ratio}")
                    else:
                        tqdm.write("Error: Test ratio must be between 0 and 1.")
                except ValueError:
                    tqdm.write("Error: Test ratio must be a number.")
        
        elif choice == "back" or choice is None:
            break
    
    return config

# Module-level functions for multiprocessing
def process_api_specification(file_path: str, output_path: str) -> bool:
    """
    Process an API specification file (OpenAPI/Swagger) into a structured format.
    
    Args:
        file_path (str): Path to the API specification file
        output_path (str): Path where the Markdown file will be saved
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            spec = json.load(f)
        
        # Transform to markdown with structured API info
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# API Reference: {spec.get('info', {}).get('title', 'Unnamed API')}\n\n")
            f.write(f"Version: {spec.get('info', {}).get('version', 'Unknown')}\n\n")
            
            if 'info' in spec and 'description' in spec['info']:
                f.write(f"{spec['info']['description']}\n\n")
            
            # Extract schemas/definitions
            schema_section = None
            if 'components' in spec and 'schemas' in spec['components']:
                schema_section = spec['components']['schemas']
            elif 'definitions' in spec:  # For older Swagger specs
                schema_section = spec['definitions']
            
            if schema_section:
                f.write("## Data Models\n\n")
                for schema_name, schema_def in schema_section.items():
                    f.write(f"### {schema_name}\n\n")
                    if 'description' in schema_def:
                        f.write(f"{schema_def['description']}\n\n")
                    
                    if 'properties' in schema_def:
                        f.write("**Properties:**\n\n")
                        for prop_name, prop_details in schema_def['properties'].items():
                            prop_type = prop_details.get('type', 'object')
                            if prop_type == 'array' and 'items' in prop_details:
                                items_type = prop_details['items'].get('type', 'object')
                                if '$ref' in prop_details['items']:
                                    ref = prop_details['items']['$ref'].split('/')[-1]
                                    items_type = f"[{ref}]"
                                prop_type = f"array of {items_type}"
                            elif '$ref' in prop_details:
                                ref = prop_details['$ref'].split('/')[-1]
                                prop_type = f"[{ref}]"
                            
                            f.write(f"- `{prop_name}` ({prop_type}): {prop_details.get('description', '')}\n")
                        f.write("\n")
            
            # Extract endpoints
            if 'paths' in spec:
                f.write("## Endpoints\n\n")
                for path, methods in spec['paths'].items():
                    for method, details in methods.items():
                        if method in ['get', 'post', 'put', 'delete', 'patch']:
                            f.write(f"### `{method.upper()} {path}`\n\n")
                            f.write(f"{details.get('summary', '')}\n\n")
                            f.write(f"{details.get('description', '')}\n\n")
                            
                            # Parameters
                            if 'parameters' in details and details['parameters']:
                                f.write("**Parameters:**\n\n")
                                for param in details['parameters']:
                                    param_type = param.get('schema', {}).get('type', param.get('type', 'unknown'))
                                    required = "required" if param.get('required', False) else "optional"
                                    f.write(f"- `{param.get('name')}` ({param.get('in', 'unknown')}, {param_type}, {required}): {param.get('description', '')}\n")
                                f.write("\n")
                            
                            # Request body
                            if 'requestBody' in details:
                                f.write("**Request Body:**\n\n")
                                content = details['requestBody'].get('content', {})
                                for content_type, content_details in content.items():
                                    f.write(f"Content-Type: `{content_type}`\n\n")
                                    if 'schema' in content_details:
                                        schema = content_details['schema']
                                        if '$ref' in schema:
                                            ref = schema['$ref'].split('/')[-1]
                                            f.write(f"Schema: `{ref}`\n\n")
                                        else:
                                            f.write("```json\n")
                                            f.write(json.dumps(schema, indent=2))
                                            f.write("\n```\n\n")
                            
                            # Responses
                            if 'responses' in details:
                                f.write("**Responses:**\n\n")
                                for status, response in details['responses'].items():
                                    f.write(f"- `{status}`: {response.get('description', '')}\n")
                                    if 'content' in response:
                                        for content_type, content_details in response['content'].items():
                                            if 'schema' in content_details:
                                                schema = content_details['schema']
                                                if '$ref' in schema:
                                                    ref = schema['$ref'].split('/')[-1]
                                                    f.write(f"  Schema: `{ref}`\n")
                                f.write("\n")
                                
                            # Examples if available
                            if 'examples' in details:
                                f.write("**Examples:**\n\n")
                                for example_name, example in details['examples'].items():
                                    f.write(f"Example: {example_name}\n\n")
                                    if 'value' in example:
                                        f.write("```json\n")
                                        f.write(json.dumps(example['value'], indent=2))
                                        f.write("\n```\n\n")
        
        return True
    except Exception as e:
        logger.error(f"Error processing API spec {file_path}: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

def extract_function_signature(py_path: str) -> List[Dict[str, Any]]:
    """
    Extract detailed function signatures from a Python file.
    
    Args:
        py_path (str): Path to the Python file
        
    Returns:
        List[Dict[str, Any]]: List of function metadata dictionaries
    """
    try:
        with open(py_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        functions = []
        
        # Find function definitions with potential return type annotations
        function_matches = list(re.finditer(
            r'def\s+([a-zA-Z0-9_]+)\s*\((.*?)\)(?:\s*->\s*([a-zA-Z0-9_\[\]\.\'\"<>, ]+))?\s*:', 
            content, 
            re.DOTALL
        ))
        
        for match in function_matches:
            func_name = match.group(1)
            func_params_raw = match.group(2).strip()
            return_type = match.group(3).strip() if match.group(3) else None
            
            # Parse parameters with their type annotations
            params = []
            if func_params_raw:
                # Split by commas but handle nested types with commas like List[int, str]
                param_parts = []
                current_part = ""
                bracket_depth = 0
                
                for char in func_params_raw:
                    if char == '[':
                        bracket_depth += 1
                    elif char == ']':
                        bracket_depth -= 1
                    
                    if char == ',' and bracket_depth == 0:
                        param_parts.append(current_part.strip())
                        current_part = ""
                    else:
                        current_part += char
                
                if current_part:
                    param_parts.append(current_part.strip())
                
                for param in param_parts:
                    # Check for parameter with type annotation
                    type_annotation_match = re.match(r'([a-zA-Z0-9_]+)\s*:\s*([a-zA-Z0-9_\[\]\.\'\"<>, ]+)(?:\s*=\s*(.+))?', param)
                    if type_annotation_match:
                        param_name = type_annotation_match.group(1)
                        param_type = type_annotation_match.group(2)
                        default_value = type_annotation_match.group(3) if type_annotation_match.group(3) else None
                        params.append({
                            'name': param_name,
                            'type': param_type,
                            'default': default_value
                        })
                    else:
                        # Check for parameter with default value
                        default_match = re.match(r'([a-zA-Z0-9_]+)(?:\s*=\s*(.+))?', param)
                        if default_match:
                            param_name = default_match.group(1)
                            default_value = default_match.group(2) if default_match.group(2) else None
                            params.append({
                                'name': param_name,
                                'type': None,
                                'default': default_value
                            })
            
            # Extract docstring
            func_start = match.end()
            next_def = content.find('def ', func_start)
            if next_def == -1:
                next_def = len(content)
            
            func_body = content[func_start:next_def].strip()
            docstring_match = re.search(r'"""(.*?)"""', func_body, re.DOTALL)
            docstring = docstring_match.group(1).strip() if docstring_match else ""
            
            # Extract parameter info from docstring
            param_info = {}
            param_section_match = re.search(r'(?:Args|Parameters):(.*?)(?:Returns:|Raises:|Yields:|Examples:|$)', docstring, re.DOTALL)
            if param_section_match:
                param_section = param_section_match.group(1).strip()
                # Extract individual parameter descriptions
                param_pattern = re.compile(r'([a-zA-Z0-9_]+)(?:\s*\((.*?)\))?\s*:(.*?)(?=\n\s*[a-zA-Z0-9_]+\s*(?:\(.*?\))?\s*:|$)', re.DOTALL)
                for param_match in param_pattern.finditer(param_section):
                    param_name = param_match.group(1)
                    param_type = param_match.group(2) if param_match.group(2) else None
                    param_desc = param_match.group(3).strip()
                    param_info[param_name] = {'type': param_type, 'description': param_desc}
            
            # Extract return info from docstring
            return_info = {}
            return_section_match = re.search(r'Returns:(.*?)(?:Raises:|Yields:|Examples:|$)', docstring, re.DOTALL)
            if return_section_match:
                return_section = return_section_match.group(1).strip()
                # Check if there's a type specification
                return_type_match = re.match(r'([a-zA-Z0-9_\[\]\.\'\"<>, ]+):\s*(.*)', return_section, re.DOTALL)
                if return_type_match:
                    return_info = {
                        'type': return_type_match.group(1).strip(),
                        'description': return_type_match.group(2).strip()
                    }
                else:
                    return_info = {'description': return_section}
            
            # Extract examples from docstring
            examples = []
            examples_match = re.search(r'Examples?(.*?)(?:$)', docstring, re.DOTALL)
            if examples_match:
                examples_section = examples_match.group(1).strip()
                # Look for code blocks
                code_blocks = re.findall(r'```(?:python)?\s*(.*?)\s*```', examples_section, re.DOTALL)
                if code_blocks:
                    examples.extend(code_blocks)
                else:
                    # Look for indented code examples
                    lines = examples_section.split('\n')
                    current_example = []
                    in_example = False
                    
                    for line in lines:
                        if line.strip() and line.startswith('    '):  # 4-space indent
                            in_example = True
                            current_example.append(line.strip())
                        elif in_example and not line.strip():  # Empty line in example
                            current_example.append('')
                        elif in_example:  # End of example
                            examples.append('\n'.join(current_example))
                            current_example = []
                            in_example = False
                    
                    if current_example:  # Add the last example if there is one
                        examples.append('\n'.join(current_example))
            
            # Extract full function definition
            func_def_start = content.rfind('\n', 0, match.start())
            if func_def_start == -1:
                func_def_start = 0
            else:
                func_def_start += 1  # Skip the newline
            
            func_def_end = content.find('\n', match.end())
            if func_def_end == -1:
                func_def_end = len(content)
            
            # Include any decorators
            decorator_starts = []
            decorator_pattern = re.compile(r'^\s*@[a-zA-Z0-9_\.]+.*$', re.MULTILINE)
            for decorator_match in decorator_pattern.finditer(content[:func_def_start]):
                decorator_line_end = content.find('\n', decorator_match.start())
                if decorator_line_end != -1 and decorator_line_end < func_def_start:
                    next_non_decorator = content.find('\n', decorator_line_end + 1)
                    while next_non_decorator != -1 and re.match(r'^\s*@[a-zA-Z0-9_\.]+.*$', content[decorator_line_end+1:next_non_decorator]):
                        decorator_line_end = next_non_decorator
                        next_non_decorator = content.find('\n', decorator_line_end + 1)
                    
                    if func_def_start - decorator_match.start() < 200:  # Only include if close to function
                        func_def_start = decorator_match.start()
            
            full_definition = content[func_def_start:func_def_end].strip()
            
            functions.append({
                'name': func_name,
                'parameters': params,
                'return_type': return_type,
                'docstring': docstring,
                'parameter_info': param_info,
                'return_info': return_info,
                'examples': examples,
                'full_definition': full_definition
            })
        
        return functions
    except Exception as e:
        logger.error(f"Error extracting function signatures from {py_path}: {str(e)}")
        logger.debug(traceback.format_exc())
        return []

def convert_py_to_md_enhanced(py_path: str, output_path: str) -> bool:
    """
    Convert a Python file to Markdown format with enhanced code extraction for SDK documentation.
    
    Args:
        py_path (str): Path to the Python file.
        output_path (str): Path where the Markdown file will be saved.
        
    Returns:
        bool: True if conversion was successful, False otherwise.
    """
    try:
        with open(py_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Extract module docstring
        module_docstring = re.search(r'"""(.*?)"""', content, re.DOTALL)
        
        # Extract import statements
        imports = re.findall(r'^(?:from|import)\s+.*?$', content, re.MULTILINE)
        
        # Extract class definitions
        classes = list(re.finditer(r'class\s+([a-zA-Z0-9_]+)(?:\(([a-zA-Z0-9_., \[\]\'\"]+)\))?:(.*?)(?=class|\Z)', content, re.DOTALL))
        
        # Extract function signatures
        functions = extract_function_signature(py_path)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# {os.path.basename(py_path)}\n\n")
            
            # Add SDK identification
            f.write("## SDK Documentation\n\n")
            f.write(f"This file contains documentation for `{os.path.basename(py_path)}`.\n\n")

            if module_docstring:
                f.write("## Module Description\n\n")
                f.write(module_docstring.group(1).strip() + "\n\n")
            
            # Write imports section
            if imports:
                f.write("## Imports\n\n")
                f.write("```python\n")
                f.write("\n".join(imports))
                f.write("\n```\n\n")
            
            # Write classes
            if classes:
                f.write("## Classes\n\n")
                for cls in classes:
                    cls_name = cls.group(1)
                    cls_parents = cls.group(2) if cls.group(2) else ""
                    cls_body = cls.group(3)

                    # Extract class docstring
                    cls_docstring = re.search(r'"""(.*?)"""', cls_body, re.DOTALL)
                    
                    # Get class methods
                    cls_methods = list(re.finditer(r'def\s+([a-zA-Z0-9_]+)\s*\((.*?)\)(?:\s*->\s*([a-zA-Z0-9_\[\]\.\'\"<>, ]+))?\s*:', cls_body, re.DOTALL))
                    
                    f.write(f"### `{cls_name}`\n\n")
                    if cls_parents:
                        f.write(f"Inherits from: `{cls_parents}`\n\n")
                    
                    if cls_docstring:
                        f.write(cls_docstring.group(1).strip() + "\n\n")
                    
                    # Write class methods
                    if cls_methods:
                        f.write(f"#### Methods\n\n")
                        for method in cls_methods:
                            method_name = method.group(1)
                            method_params = method.group(2)
                            method_return = method.group(3) if method.group(3) else ""
                            
                            method_signature = f"{method_name}({method_params})"
                            if method_return:
                                method_signature += f" -> {method_return}"
                            
                            f.write(f"##### `{method_signature}`\n\n")
                            
                            # Extract method docstring
                            method_body = cls_body[method.start():method.end() + 200]  # Add some context
                            method_docstring = re.search(r'"""(.*?)"""', method_body, re.DOTALL)
                            
                            if method_docstring:
                                f.write(method_docstring.group(1).strip() + "\n\n")
            
            # Write functions
            if functions:
                f.write("## Functions\n\n")
                for func in functions:
                    # Create clean signature for display
                    params_str = []
                    for param in func['parameters']:
                        param_str = param['name']
                        if param['type']:
                            param_str += f": {param['type']}"
                        if param['default']:
                            param_str += f" = {param['default']}"
                        params_str.append(param_str)
                    
                    signature = f"{func['name']}({', '.join(params_str)})"
                    if func['return_type']:
                        signature += f" -> {func['return_type']}"
                    
                    f.write(f"### `{signature}`\n\n")
                    
                    if func['docstring']:
                        f.write(func['docstring'] + "\n\n")
                    
                    # Add parameter table if we have detailed info
                    if func['parameter_info']:
                        f.write("**Parameters:**\n\n")
                        f.write("| Name | Type | Description |\n")
                        f.write("|------|------|-------------|\n")
                        
                        for param in func['parameters']:
                            param_name = param['name']
                            param_type = param['type'] or ''
                            
                            # Try to get type from docstring if not in signature
                            if not param_type and param_name in func['parameter_info']:
                                param_type = func['parameter_info'][param_name].get('type', '')
                            
                            param_desc = ''
                            if param_name in func['parameter_info']:
                                param_desc = func['parameter_info'][param_name].get('description', '')
                            
                            f.write(f"| `{param_name}` | `{param_type}` | {param_desc} |\n")
                        
                        f.write("\n")
                    
                    # Add return info
                    if func['return_info']:
                        f.write("**Returns:**\n\n")
                        if 'type' in func['return_info'] and func['return_info']['type']:
                            f.write(f"Type: `{func['return_info']['type']}`\n\n")
                        f.write(f"{func['return_info'].get('description', '')}\n\n")
                    
                    # Add examples
                    if func['examples']:
                        f.write("**Examples:**\n\n")
                        for example in func['examples']:
                            f.write("```python\n")
                            f.write(example)
                            f.write("\n```\n\n")
                    
                    # Add full definition for reference
                    f.write("**Definition:**\n\n")
                    f.write("```python\n")
                    f.write(func['full_definition'])
                    f.write("\n```\n\n")
            
            # Add original code as a code block at the end
            f.write("## Original Code\n\n```python\n")
            f.write(content)
            f.write("\n```\n")

        return True
    except Exception as e:
        logger.error(f"Error converting Python file {py_path} to markdown: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

def convert_ipynb_to_md(notebook_path: str, output_path: str) -> bool:
    """
    Convert a Jupyter notebook to Markdown format.

    Args:
        notebook_path (str): Path to the Jupyter notebook file.
        output_path (str): Path where the Markdown file will be saved.

    Returns:
        bool: True if conversion was successful, False otherwise.
    """
    try:
        # Suppress nbformat validation warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            with open(notebook_path, 'r', encoding='utf-8', errors='ignore') as f:
                notebook = nbformat.read(f, as_version=4)

            # Skip validation and adding IDs altogether since it's causing warnings
            exporter = MarkdownExporter()
            (body, resources) = exporter.from_notebook_node(notebook)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"# {os.path.basename(notebook_path)}\n\n")
                f.write(body)

            return True
    except Exception as e:
        logger.error(f"Error converting notebook {notebook_path} to markdown: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

def convert_py_to_md(py_path: str, output_path: str) -> bool:
    """
    Convert a Python file to Markdown format, extracting docstrings and comments.

    Args:
        py_path (str): Path to the Python file.
        output_path (str): Path where the Markdown file will be saved.

    Returns:
        bool: True if conversion was successful, False otherwise.
    """
    try:
        with open(py_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Extract module docstring
        module_docstring = re.search(r'"""(.*?)"""', content, re.DOTALL)

        # Extract functions and their docstrings
        functions = list(re.finditer(r'def\s+([a-zA-Z0-9_]+)\s*\((.*?)\):(.*?)(?=def|\Z)', content, re.DOTALL))

        # Extract classes and their docstrings
        classes = list(re.finditer(r'class\s+([a-zA-Z0-9_]+).*?:(.*?)(?=class|\Z)', content, re.DOTALL))

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# {os.path.basename(py_path)}\n\n")

            if module_docstring:
                f.write("## Module Description\n\n")
                f.write(module_docstring.group(1).strip() + "\n\n")

            # Write functions
            if functions:
                f.write("## Functions\n\n")
                for func in functions:
                    func_name = func.group(1)
                    func_params = func.group(2)
                    func_body = func.group(3)

                    # Extract function docstring
                    func_docstring = re.search(r'"""(.*?)"""', func_body, re.DOTALL)

                    f.write(f"### `{func_name}({func_params})`\n\n")
                    if func_docstring:
                        f.write(func_docstring.group(1).strip() + "\n\n")
                    else:
                        # Look for single-line comments
                        comments = re.findall(r'#\s*(.*?)$', func_body, re.MULTILINE)
                        if comments:
                            f.write("\n".join(comments) + "\n\n")

            # Write classes
            if classes:
                f.write("## Classes\n\n")
                for cls in classes:
                    cls_name = cls.group(1)
                    cls_body = cls.group(2)

                    # Extract class docstring
                    cls_docstring = re.search(r'"""(.*?)"""', cls_body, re.DOTALL)

                    f.write(f"### `{cls_name}`\n\n")
                    if cls_docstring:
                        f.write(cls_docstring.group(1).strip() + "\n\n")
                    else:
                        # Look for single-line comments
                        comments = re.findall(r'#\s*(.*?)$', cls_body, re.MULTILINE)
                        if comments:
                            f.write("\n".join(comments) + "\n\n")

            # Add original code as a code block at the end
            f.write("## Original Code\n\n```python\n")
            f.write(content)
            f.write("\n```\n")

        return True
    except Exception as e:
        logger.error(f"Error converting Python file {py_path} to markdown: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

def get_file_metadata(file_path: str, repo_name: str, category: str) -> Dict[str, Any]:
    """
    Extract metadata for a given file.

    Args:
        file_path (str): Path to the file.
        repo_name (str): Name of the repository containing the file.
        category (str): Category of the file (documentation, examples, etc.)

    Returns:
        Dict[str, Any]: Dictionary containing file metadata.
    """
    try:
        stats = os.stat(file_path)
        file_ext = os.path.splitext(file_path)[1].lower()
        file_name = os.path.basename(file_path)

        # Read the file content
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Extract title from the content (first heading or filename if no heading found)
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        title = title_match.group(1) if title_match else file_name

        # Extract description (first paragraph after title or first 200 chars)
        desc_match = re.search(r'^#.*?\n+(.+?)(\n\n|\n#|\Z)', content, re.DOTALL)
        description = desc_match.group(1).strip() if desc_match else content[:200].strip()

        return {
            'file_name': file_name,
            'file_path': file_path,
            'repo_name': repo_name,
            'category': category,
            'title': title,
            'description': description,
            'file_type': file_ext[1:] if file_ext else 'unknown',
            'file_size': stats.st_size,
            'creation_date': datetime.datetime.fromtimestamp(stats.st_ctime).isoformat(),
            'modification_date': datetime.datetime.fromtimestamp(stats.st_mtime).isoformat(),
            'content': content,
        }
    except Exception as e:
        logger.error(f"Error getting metadata for {file_path}: {str(e)}")
        return {
            'file_name': os.path.basename(file_path),
            'file_path': file_path,
            'repo_name': repo_name,
            'category': category,
            'title': os.path.basename(file_path),
            'description': '',
            'file_type': 'unknown',
            'file_size': 0,
            'creation_date': '',
            'modification_date': '',
            'content': '',
        }

# Functions for multiprocessing
def convert_file(args):
    """
    Convert a file based on its type.
    
    Args:
        args: Tuple containing (file_path, output_path, file_type)
        
    Returns:
        bool: True if conversion was successful, False otherwise
    """
    file_path, output_path, file_type = args
    
    try:
        if file_type == 'ipynb':
            return convert_ipynb_to_md(file_path, output_path)
        elif file_type == 'py':
            return convert_py_to_md_enhanced(file_path, output_path)
        else:
            # For other file types, just copy
            shutil.copy2(file_path, output_path)
            return True
    except Exception as e:
        logger.error(f"Error converting file {file_path}: {str(e)}")
        return False

# Function for parallel processing of files
def process_file(args) -> Optional[Dict[str, Any]]:
    """
    Process a single file - convert and extract metadata.
    
    Args:
        args: Tuple containing (src_file, dest_dir, category, repo_name, rel_within_category)
        
    Returns:
        Dict[str, Any]: File metadata or None if processing failed
    """
    src_file, dest_dir, category, repo_name, rel_within_category = args
    
    try:
        # Create the destination directory
        dest_dir_path = os.path.join(dest_dir, category, os.path.dirname(rel_within_category))
        os.makedirs(dest_dir_path, exist_ok=True)

        # Process based on file type
        file_base, file_ext = os.path.splitext(os.path.basename(src_file))
        file_ext = file_ext.lower()

        if file_ext == '.json':
            # Check if it's an API specification
            try:
                with open(src_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    data = json.loads(content)
                
                # If it looks like an OpenAPI spec, process it specially
                if isinstance(data, dict) and any(key in data for key in ['swagger', 'openapi', 'paths']):
                    md_filename = f"{file_base}_api_spec.md"
                    dest_file = os.path.join(dest_dir_path, md_filename)
                    if process_api_specification(src_file, dest_file):
                        metadata = get_file_metadata(dest_file, repo_name, category)
                        metadata['is_api_spec'] = True
                        return standardize_metadata(metadata)
                else:
                    # Regular JSON file - copy it
                    dest_file = os.path.join(dest_dir_path, os.path.basename(src_file))
                    shutil.copy2(src_file, dest_file)
                    metadata = get_file_metadata(dest_file, repo_name, category)
                    return standardize_metadata(metadata)
            except Exception as e:
                logger.warning(f"Error processing JSON file {src_file}: {str(e)}")
                # If we can't parse as JSON, skip it
                return None
        
        elif file_ext == '.ipynb':
            # Convert notebook to markdown
            md_filename = f"{file_base}.md"
            dest_file = os.path.join(dest_dir_path, md_filename)
            if convert_ipynb_to_md(src_file, dest_file):
                metadata = get_file_metadata(dest_file, repo_name, category)
                return standardize_metadata(metadata)

        elif file_ext == '.py':
            # Convert Python file to enhanced markdown with SDK info
            md_filename = f"{file_base}.md"
            dest_file = os.path.join(dest_dir_path, md_filename)
            if convert_py_to_md_enhanced(src_file, dest_file):
                metadata = get_file_metadata(dest_file, repo_name, category)
                metadata['is_sdk_file'] = True
                return standardize_metadata(metadata)

        elif file_ext in ['.md', '.rst', '.txt']:
            # Copy other documentation file types as is
            dest_file = os.path.join(dest_dir_path, os.path.basename(src_file))
            try:
                shutil.copy2(src_file, dest_file)
                metadata = get_file_metadata(dest_file, repo_name, category)
                return standardize_metadata(metadata)
            except Exception as e:
                logger.warning(f"Error copying {src_file}: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error processing file {src_file}: {str(e)}")
    
    return None

# Function for parallel dataset processing
def process_file_for_dataset(args):
    """
    Process a file for inclusion in the dataset.
    
    Args:
        args: Tuple containing (file_path, repo_name, category, split)
        
    Returns:
        Dict[str, Any]: File metadata with split information or None if not a documentation file
    """
    file_path, repo_name, category, split = args
    
    try:
        # Check if it's a documentation file
        if not is_documentation_file(file_path):
            return None
            
        # Get metadata
        metadata = get_file_metadata(file_path, repo_name, category)
        metadata['split'] = split
        
        # Standardize metadata
        return standardize_metadata(metadata)
    except Exception as e:
        logger.error(f"Error processing file {file_path} for dataset: {str(e)}")
        return None

# Enhanced Functions with Progress Bars and Multiprocessing

def get_organization_repos_hf(organization: str, token: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieve all repositories for a given organization from Hugging Face Hub.

    Args:
        organization (str): The name of the organization on Hugging Face Hub.
        token (Optional[str]): Hugging Face API token.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing repository information.
    """
    try:
        api = HfApi(token=token)

        # Try several different approaches to find repositories
        approaches = [
            ("list_models", lambda: list(api.list_models(author=organization))),
            ("list_datasets", lambda: list(api.list_datasets(author=organization))),
            ("direct API request", lambda: requests.get(
                f"https://huggingface.co/api/models?author={organization}",
                headers={"Authorization": f"Bearer {token}"} if token else None
            ).json() if requests.get(f"https://huggingface.co/api/models?author={organization}").status_code == 200 else []),
            ("search", lambda: [repo for repo in list(api.list_models(search=organization)) 
                               if organization.lower() in repo.id.lower()])
        ]
        
        with tqdm(total=len(approaches), desc=f"Searching repositories for {organization}") as pbar:
            for method_name, method_func in approaches:
                try:
                    pbar.set_description(f"Trying {method_name}")
                    repos = method_func()
                    if repos:
                        tqdm.write(f"Found {len(repos)} repositories for {organization} using {method_name}.")
                        return repos
                except Exception as e:
                    tqdm.write(f"Warning: {method_name} approach failed: {str(e)}")
                finally:
                    pbar.update(1)
        
        tqdm.write(f"No repositories found for {organization} on Hugging Face Hub")
        return []
    except Exception as e:
        tqdm.write(f"Error retrieving repositories for {organization} from Hugging Face: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def get_organization_repos_github(organization: str, token: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieve all repositories for a given organization from GitHub.

    Args:
        organization (str): The name of the organization on GitHub.
        token (Optional[str]): GitHub API token for authentication.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing repository information.
    """
    try:
        # GitHub API request with authentication if token provided
        url = f"https://api.github.com/orgs/{organization}/repos?per_page=100"
        headers = {"Authorization": f"token {token}"} if token else {}
        repos = []
        page = 1
        
        with tqdm(desc=f"Fetching GitHub repositories for {organization}", unit="page") as pbar:
            while True:
                response = requests.get(f"{url}&page={page}", headers=headers)
                
                if response.status_code == 403 and 'rate limit exceeded' in response.text.lower():
                    tqdm.write("GitHub API rate limit exceeded. Please provide a GitHub token.")
                    break
                
                if response.status_code != 200:
                    tqdm.write(f"GitHub API returned status code {response.status_code}")
                    break

                page_repos = response.json()
                if not page_repos:
                    break

                repos.extend(page_repos)
                pbar.update(1)
                pbar.set_description(f"Fetched {len(repos)} repositories (page {page})")
                page += 1

                # Check if we've reached the last page
                if 'next' not in response.links:
                    break

        tqdm.write(f"Found {len(repos)} repositories for {organization} on GitHub.")
        return repos
    except Exception as e:
        tqdm.write(f"Error retrieving repositories for {organization} from GitHub: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def download_repo_content_hf(repo_id: str, output_dir: str, token: Optional[str] = None) -> Optional[str]:
    """
    Download the repository content from Hugging Face to a temporary directory.

    Args:
        repo_id (str): The ID of the repository (e.g., "organization/repo-name").
        output_dir (str): Directory where the repository will be cloned.
        token (Optional[str]): Hugging Face API token.

    Returns:
        Optional[str]: Path to the downloaded repository or None if download failed.
    """
    try:
        tqdm.write(f"Downloading Hugging Face repository: {repo_id}")
        repo_path = os.path.join(output_dir, repo_id.split('/')[-1])
        os.makedirs(repo_path, exist_ok=True)

        # Download the repository content with progress tracking
        with tqdm(desc=f"Downloading {repo_id}", unit="files") as pbar:
            snapshot_download(
                repo_id=repo_id,
                local_dir=repo_path,
                repo_type="model",
                token=token,
                ignore_patterns=["*.bin", "*.pt", "*.pth", "*.ckpt", "*.safetensors", ".git*"],
                ignore_errors=True  # Continue even if some files fail to download
            )

        tqdm.write(f"Successfully downloaded {repo_id} to {repo_path}")
        return repo_path
    except Exception as e:
        tqdm.write(f"Error downloading Hugging Face repository {repo_id}: {str(e)}")
        return None

def download_repo_content_github(repo_url: str, output_dir: str, token: Optional[str] = None) -> Optional[str]:
    """
    Download the repository content from GitHub to a temporary directory.

    Args:
        repo_url (str): The URL of the GitHub repository.
        output_dir (str): Directory where the repository will be cloned.
        token (Optional[str]): GitHub API token.

    Returns:
        Optional[str]: Path to the downloaded repository or None if download failed.
    """
    try:
        repo_name = repo_url.split('/')[-1]
        tqdm.write(f"Cloning GitHub repository: {repo_url}")
        repo_path = os.path.join(output_dir, repo_name)
        
        # If token is provided, use it in the URL
        clone_url = repo_url
        if token:
            # Parse the URL to add the token
            url_parts = repo_url.split('://')
            if len(url_parts) == 2:
                clone_url = f"{url_parts[0]}://{token}@{url_parts[1]}"
        
        # Custom progress class to track git clone progress
        class GitProgressBar(git.RemoteProgress):
            def __init__(self):
                super().__init__()
                self.pbar = tqdm(desc=f"Cloning {repo_name}", unit="objects")
            
            def update(self, op_code, cur_count, max_count=None, message=''):
                if max_count:
                    self.pbar.total = max_count
                self.pbar.n = cur_count
                self.pbar.refresh()
        
        # Clone the repository with progress
        progress_bar = GitProgressBar()
        git.Repo.clone_from(clone_url, repo_path, progress=progress_bar)

        tqdm.write(f"Successfully cloned {repo_url} to {repo_path}")
        return repo_path
    except Exception as e:
        tqdm.write(f"Error cloning GitHub repository {repo_url}: {str(e)}")
        return None

def find_files_in_target_folders(repo_path: str, file_types: List[str], target_folders: List[str]) -> Dict[str, List[str]]:
    """
    Find files with the specified extensions only within the target folders.

    Args:
        repo_path (str): Path to the repository.
        file_types (List[str]): List of file extensions to look for.
        target_folders (List[str]): List of folder names to search in.

    Returns:
        Dict[str, List[str]]: Dictionary with folder categories as keys and lists of file paths as values.
    """
    result = {folder: [] for folder in target_folders}
    result['other'] = []  # For files that match but are in non-standard folders

    # Count total directories for progress bar
    total_dirs = sum(1 for _ in os.walk(repo_path))
    
    # First, find all target folder paths in the repository
    target_folder_paths = []
    
    with tqdm(total=total_dirs, desc=f"Scanning {os.path.basename(repo_path)} directories", leave=False) as pbar:
        for root, dirs, _ in os.walk(repo_path):
            pbar.update(1)
            
            # Skip .git directory
            if '.git' in root.split(os.path.sep):
                continue

            for dir_name in dirs:
                if dir_name.lower() in [f.lower() for f in target_folders]:
                    target_folder_paths.append((os.path.join(root, dir_name), dir_name))
    
    tqdm.write(f"Found {len(target_folder_paths)} target directories to scan")

    # Now scan only within those target folders
    all_files_count = 0
    
    with tqdm(total=len(target_folder_paths), desc="Scanning target folders", leave=False) as pbar:
        for folder_path, original_folder_name in target_folder_paths:
            pbar.set_description(f"Scanning {original_folder_name}")
            
            # Find the matching target folder name (preserving case)
            target_category = next((t for t in target_folders if t.lower() == original_folder_name.lower()), 'other')

            # Find matching files in this folder and its subdirectories
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if any(file.endswith(ext) for ext in file_types):
                        file_path = os.path.join(root, file)
                        result[target_category].append(file_path)
                        all_files_count += 1
            
            pbar.update(1)

    # Log the findings
    tqdm.write(f"Found {all_files_count} matching files in target folders")

    # Log counts by category
    for category, files in result.items():
        if files:
            tqdm.write(f"  - {category}: {len(files)} files")

    return result

def extract_and_convert_files(repo_path: str, target_dir: str, repo_name: str) -> Dict[str, List[str]]:
    """
    Extract and convert files from target folders in a repository.

    Args:
        repo_path (str): Path to the downloaded repository.
        target_dir (str): Directory where extracted files will be copied.
        repo_name (str): Name of the repository for organizing purposes.

    Returns:
        Dict[str, List[str]]: Dictionary with keys as file categories and values as lists of extracted file paths.
    """
    # Define interesting file types and target folders - INCLUDE ALL TYPES FOR SDK DOCS
    file_types = ['.md', '.rst', '.txt', '.py', '.ipynb', '.json']
    target_folders = ['docs', 'examples', 'cookbook', 'cookbooks', 'documentation', 'tutorials']

    # Create target directory for this repo
    repo_target_dir = os.path.join(target_dir, repo_name)
    os.makedirs(repo_target_dir, exist_ok=True)

    # Create directories for each category
    for category in target_folders + ['other']:
        os.makedirs(os.path.join(repo_target_dir, category), exist_ok=True)

    try:
        # Find files in target folders
        files_by_category = find_files_in_target_folders(repo_path, file_types, target_folders)

        # Track extracted files
        extracted_files = {category: [] for category in files_by_category.keys()}
        all_file_metadatas = []
        
        # Prepare file processing tasks
        processing_tasks = []
        
        # Prepare tasks for multiprocessing
        for category, files in files_by_category.items():
            for src_file in files:
                # Determine the relative path to preserve structure within the category
                # First get the path relative to the repo, then relative to the category folder
                rel_to_repo = os.path.relpath(src_file, repo_path)

                # Find which part of the path contains the category folder
                parts = rel_to_repo.split(os.path.sep)
                category_index = -1
                for i, part in enumerate(parts):
                    if part.lower() == category.lower():
                        category_index = i
                        break

                # Get the part of the path after the category folder
                if category_index >= 0 and category_index < len(parts) - 1:
                    rel_within_category = os.path.join(*parts[category_index+1:])
                else:
                    # If category folder is not in the path, use the filename
                    rel_within_category = os.path.basename(src_file)
                
                # Add task to processing queue
                processing_tasks.append((src_file, repo_target_dir, category, repo_name, rel_within_category))
        
        # Process files in parallel using multiprocessing
        total_files = len(processing_tasks)
        if total_files > 0:
            # Calculate optimal number of workers (leave 1 CPU for system)
            num_workers = max(1, min(multiprocessing.cpu_count() - 1, 8))
            
            with tqdm(total=total_files, desc=f"Processing {repo_name} files", leave=True) as pbar:
                # Use process pool for parallel processing
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = [executor.submit(process_file, task) for task in processing_tasks]
                    
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            if result:  # Only add non-None results
                                category = result.get('category', 'other')
                                extracted_files[category].append(result.get('file_path'))
                                all_file_metadatas.append(result)
                        except Exception as e:
                            logger.error(f"Error in file processing task: {str(e)}")
                        
                        pbar.update(1)
        
        tqdm.write(f"Extracted and converted files from {repo_name}")

        # Count files by category
        for category, files in extracted_files.items():
            if files:
                tqdm.write(f"  - {category}: {len(files)} files")

        return extracted_files
    except Exception as e:
        tqdm.write(f"Error extracting files from {repo_name}: {str(e)}")
        logger.error(traceback.format_exc())
        return {folder: [] for folder in target_folders + ['other']}

def split_into_train_test(files_by_category: Dict[str, List[str]], test_ratio: float = 0.2) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split files into training and testing sets while preserving category information.

    Args:
        files_by_category (Dict[str, List[str]]): Dictionary of files by category.
        test_ratio (float): Ratio of files to use for testing (default: 0.2).

    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: Training and testing metadata.
    """
    train_data = []
    test_data = []

    # Prepare tasks for parallel processing
    all_tasks = []
    
    for category, files in files_by_category.items():
        if not files:
            continue

        # Determine repository name from the first file's path
        if files:
            repo_name = os.path.basename(os.path.dirname(os.path.dirname(files[0])))
        else:
            repo_name = "unknown"

        # Shuffle the files for this category
        random.shuffle(files)

        # Calculate the split point
        split_idx = max(1, int(len(files) * (1 - test_ratio)))

        # Split the files
        category_train_files = [(file_path, repo_name, category, 'train') for file_path in files[:split_idx]]
        category_test_files = [(file_path, repo_name, category, 'test') for file_path in files[split_idx:]]
        
        all_tasks.extend(category_train_files)
        all_tasks.extend(category_test_files)

    # Process files in parallel
    num_workers = max(1, min(multiprocessing.cpu_count() - 1, 8))
    
    with tqdm(total=len(all_tasks), desc="Preparing dataset", unit="files") as pbar:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_file_for_dataset, task) for task in all_tasks]
            
            for future in as_completed(futures):
                try:
                    metadata = future.result()
                    if metadata:  # Only add non-None results
                        if metadata['split'] == 'train':
                            train_data.append(metadata)
                        else:
                            test_data.append(metadata)
                except Exception as e:
                    logger.error(f"Error processing file metadata: {str(e)}")
                
                pbar.update(1)

    return train_data, test_data

def prepare_dataset_for_sft(dataset: DatasetDict) -> DatasetDict:
    """
    Prepare dataset for supervised fine-tuning by adding necessary columns.
    
    Args:
        dataset (DatasetDict): The original dataset
        
    Returns:
        DatasetDict: The prepared dataset for SFT
    """
    # Function to format an example for SFT
    def format_for_sft(example):
        # Create formatted text with title and content
        text = f"# {example['title']}\n\n{example['content']}"
        
        # Create minimal metadata as a string
        metadata = {
            "repo": example['repo_name'],
            "category": example['category'],
            "type": example['file_type']
        }
        
        # Convert to JSON string for metadata field
        metadata_str = json.dumps(metadata)
        
        return {
            "text": text,
            "metadata": metadata_str
        }
    
    # Apply the formatting to both splits
    with tqdm(desc="Preparing train split for SFT", total=1) as pbar:
        formatted_train = dataset['train'].map(format_for_sft)
        pbar.update(1)
    
    with tqdm(desc="Preparing test split for SFT", total=1) as pbar:
        formatted_test = dataset['test'].map(format_for_sft) if 'test' in dataset else None
        pbar.update(1)
    
    # Create a new dataset with the formatted data
    sft_dataset = DatasetDict({
        'train': formatted_train
    })
    
    if formatted_test:
        sft_dataset['test'] = formatted_test
    
    return sft_dataset

def prepare_dataset_for_code_generation(dataset: DatasetDict) -> DatasetDict:
    """
    Prepare dataset specifically for code generation by adding fields helpful for SDK compliance.
    
    Args:
        dataset (DatasetDict): The original dataset
        
    Returns:
        DatasetDict: The prepared dataset for code generation
    """
    def enhance_for_code_gen(example):
        # Initialize SDK-specific fields
        sdk_info = {
            'imports': [],
            'classes': [],
            'functions': [],
            'examples': []
        }
        
        # Extract content based on file type
        content = example['content']
        file_type = example['file_type']
        
        # Extract import statements
        import_lines = re.findall(r'^from\s+.*?\s+import\s+.*?$|^import\s+.*?$', content, re.MULTILINE)
        sdk_info['imports'].extend(import_lines)
        
        # Extract code blocks
        code_blocks = re.findall(r'```(?:python)?\s*(.*?)\s*```', content, re.DOTALL)
        
        # Process each code block to extract relevant information
        for block in code_blocks:
            # Check if it's an example (contains function/method calls)
            if re.search(r'[a-zA-Z0-9_]+\(.*?\)', block) and not block.strip().startswith('def '):
                sdk_info['examples'].append(block.strip())
            
            # Extract function definitions
            function_matches = re.finditer(r'def\s+([a-zA-Z0-9_]+)\s*\((.*?)\)(?:\s*->\s*([a-zA-Z0-9_\[\]\.\'\"<>, ]+))?\s*:', block, re.DOTALL)
            for match in function_matches:
                func_name = match.group(1)
                params = match.group(2)
                return_type = match.group(3) if match.group(3) else None
                
                # Get the full function definition
                func_start = match.start()
                next_def = block.find('\ndef ', func_start + 1)
                if next_def == -1:
                    next_def = len(block)
                
                func_def = block[func_start:next_def].strip()
                sdk_info['functions'].append(func_def)
            
            # Extract class definitions
            class_matches = re.finditer(r'class\s+([a-zA-Z0-9_]+)(?:\(([a-zA-Z0-9_., \[\]\'\"]+)\))?:', block, re.DOTALL)
            for match in class_matches:
                class_name = match.group(1)
                parents = match.group(2) if match.group(2) else None
                
                # Get the full class definition
                class_start = match.start()
                next_class = block.find('\nclass ', class_start + 1)
                if next_class == -1:
                    next_class = len(block)
                
                class_def = block[class_start:next_class].strip()
                sdk_info['classes'].append(class_def)
        
        # Extract headings to understand the structure
        headings = re.findall(r'^(#+)\s+(.+?)$', content, re.MULTILINE)
        structured_headings = []
        for level, text in headings:
            structured_headings.append({
                'level': len(level),
                'text': text.strip()
            })
        
        # Create a structured documentation string optimized for SDK learning
        sdk_doc = f"# {example['title']}\n\n"
        
        # Add metadata context
        sdk_doc += f"Repository: {example['repo_name']}\n"
        sdk_doc += f"Category: {example['category']}\n"
        sdk_doc += f"File Type: {example['file_type']}\n\n"
        
        # Add description
        sdk_doc += f"{example['description']}\n\n"
        
        # Add structured API information
        if sdk_info['imports']:
            sdk_doc += "## Imports\n\n"
            sdk_doc += "```python\n"
            sdk_doc += "\n".join(sdk_info['imports'])
            sdk_doc += "\n```\n\n"
        
        if sdk_info['classes']:
            sdk_doc += "## Classes\n\n"
            for class_def in sdk_info['classes']:
                sdk_doc += "```python\n"
                sdk_doc += class_def
                sdk_doc += "\n```\n\n"
        
        if sdk_info['functions']:
            sdk_doc += "## Functions\n\n"
            for func_def in sdk_info['functions']:
                sdk_doc += "```python\n"
                sdk_doc += func_def
                sdk_doc += "\n```\n\n"
        
        if sdk_info['examples']:
            sdk_doc += "## Examples\n\n"
            for example_code in sdk_info['examples']:
                sdk_doc += "```python\n"
                sdk_doc += example_code
                sdk_doc += "\n```\n\n"
        
        # Add full content for completeness
        sdk_doc += "## Full Content\n\n"
        sdk_doc += content
        
        # Create the enhanced example with additional fields
        enhanced = {
            "sdk_content": sdk_doc,
            "imports": sdk_info['imports'],
            "classes": sdk_info['classes'],
            "functions": sdk_info['functions'],
            "examples": sdk_info['examples'],
            "heading_structure": structured_headings,
            "text": sdk_doc  # For SFT compatibility
        }
        
        return enhanced
    
    # Apply the formatting to both splits
    with tqdm(desc="Preparing train split for code generation", total=1) as pbar:
        formatted_train = dataset['train'].map(enhance_for_code_gen)
        pbar.update(1)
    
    with tqdm(desc="Preparing test split for code generation", total=1) as pbar:
        formatted_test = dataset['test'].map(enhance_for_code_gen) if 'test' in dataset else None
        pbar.update(1)
    
    # Create a new dataset with the formatted data
    code_gen_dataset = DatasetDict({
        'train': formatted_train
    })
    
    if formatted_test:
        code_gen_dataset['test'] = formatted_test
    
    return code_gen_dataset

def display_sft_column_mapping(dataset: DatasetDict):
    """
    Display column mapping information for SFT training.
    
    Args:
        dataset (DatasetDict): The dataset to display information for
    """
    tqdm.write("\n===== Dataset Column Mapping for SFT =====")
    tqdm.write("The dataset has been prepared for supervised fine-tuning (SFT) with the following columns:")
    
    # Get the columns from the train split
    columns = list(dataset['train'].column_names)
    
    tqdm.write(f"\nFull columns available: {', '.join(columns)}")
    tqdm.write("\nFor most SFT frameworks, use the following mappings:")
    tqdm.write("  - Main content: 'content' column (contains the document text)")
    tqdm.write("  - Title/Header: 'title' column (contains the document title)")
    
    # Prepare example of how to use the dataset
    tqdm.write("\nExample usage with transformers SFT:")
    tqdm.write("""
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("your-model-name")
    tokenizer = AutoTokenizer.from_pretrained("your-model-name")
    
    # Define a formatting function to combine title and content
    def formatting_func(example):
        text = f"# {example['title']}\\n\\n{example['content']}"
        return {"text": text}
    
    # Format the dataset and set format
    formatted_dataset = dataset.map(formatting_func)
    formatted_dataset = formatted_dataset.select_columns(["text"])
    
    # Configure training
    training_args = TrainingArguments(
        output_dir="./sft-model",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_train_epochs=3,
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset["train"],
        eval_dataset=formatted_dataset["test"],
    )
    
    # Start training
    trainer.train()
    """)
    
    tqdm.write("\nThe dataset has also been automatically prepared for SFT with 'text' and 'metadata' columns.")
    tqdm.write("You can access it with: prepared_dataset = prepare_dataset_for_sft(dataset)")
    tqdm.write("===========================================\n")

def display_code_generation_info(dataset: DatasetDict):
    """
    Display information about the code generation dataset format.
    """
    tqdm.write("\n===== Code Generation Dataset Format =====")
    tqdm.write("The code generation dataset contains enhanced fields for SDK compliance:")
    
    if 'train' in dataset and len(dataset['train']) > 0:
        sample = dataset['train'][0]
        fields = list(sample.keys())
        
        tqdm.write(f"\nFields available: {', '.join(fields)}")
        tqdm.write("\nSpecial fields for code generation:")
        tqdm.write("  - sdk_content: Formatted content optimized for SDK training")
        tqdm.write("  - imports: List of import statements extracted from the content")
        tqdm.write("  - functions: List of function definitions extracted from code blocks")
        tqdm.write("  - classes: List of class definitions extracted from the content")
        tqdm.write("  - examples: List of code examples extracted from the content")
        
        tqdm.write("\nExample usage for code generation training:")
        tqdm.write("""
        # Load the code generation dataset
        from datasets import load_from_disk
        dataset = load_from_disk("./code_generation_dataset")
        
        # The dataset is already formatted with special fields for SDK compliance
        # Use the 'sdk_content' field for training, which contains structured information
        # optimized for code generation.
        
        # Example with transformers for instruction tuning:
        def format_for_instruction_tuning(example):
            instruction = "Generate code that follows the SDK documentation guidelines and patterns."
            context = example["sdk_content"]
            response = "# I'll write code following the SDK documentation"
            
            # Use appropriate formatting template for your model
            return {
                "text": f"<|user|>\\n{instruction}\\n{context}\\n<|assistant|>\\n{response}"
            }
            
        formatted_dataset = dataset.map(format_for_instruction_tuning)
        
        # For more direct code generation tasks:
        def format_completion_examples(example):
            # Find examples of function calls in the examples
            function_calls = []
            for ex in example['examples']:
                function_calls.extend(re.findall(r'([a-zA-Z0-9_]+\(.*?\))', ex))
            
            if function_calls and example['functions']:
                # Use the function definition as the prompt
                prompt = f"# Function definition:\\n\\n```python\\n{example['functions'][0]}\\n```\\n\\n# Write code that uses this function:"
                # Use the function call example as the completion
                completion = f"```python\\n{example['examples'][0]}\\n```"
                return {"prompt": prompt, "completion": completion}
            else:
                # Fallback to general documentation
                return {"prompt": example['title'], "completion": example['content'][:200]}
        
        # Create examples for training
        completion_examples = dataset.map(format_completion_examples)
        """)
        
        tqdm.write("\nThis dataset is designed to help models learn:")
        tqdm.write("  1. The correct function and class signatures from SDK documentation")
        tqdm.write("  2. How to use functions and classes according to examples")
        tqdm.write("  3. Proper import patterns for different libraries")
        tqdm.write("  4. The relationship between API specifications and code implementation")
    
    tqdm.write("===========================================\n")

def create_dataset_from_files(train_data: List[Dict[str, Any]], test_data: List[Dict[str, Any]]) -> DatasetDict:
    """
    Create a Hugging Face dataset from lists of training and testing metadata.

    Args:
        train_data (List[Dict[str, Any]]): List of training file metadata.
        test_data (List[Dict[str, Any]]): List of testing file metadata.

    Returns:
        DatasetDict: A dataset dictionary with 'train' and 'test' splits.
    """
    # Ensure we have data
    if not train_data and not test_data:
        raise ValueError("No data available to create a dataset")
    
    # Make sure all entries have standardized schema
    train_data = [standardize_metadata(item) for item in train_data]
    test_data = [standardize_metadata(item) for item in test_data]
    
    # Create the datasets with progress tracking
    with tqdm(desc="Creating training dataset", total=1) as pbar:
        train_dataset = Dataset.from_list(train_data) if train_data else Dataset.from_dict({})
        pbar.update(1)
    
    with tqdm(desc="Creating testing dataset", total=1) as pbar:
        test_dataset = Dataset.from_list(test_data) if test_data else Dataset.from_dict({})
        pbar.update(1)

    return DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })

def create_csv_files(train_data: List[Dict[str, Any]], test_data: List[Dict[str, Any]], output_dir: str):
    """
    Create CSV files from the training and testing metadata.

    Args:
        train_data (List[Dict[str, Any]]): List of training file metadata.
        test_data (List[Dict[str, Any]]): List of testing file metadata.
        output_dir (str): Directory where the CSV files will be saved.
    """
    # Define CSV fields - excluding content to keep CSV files manageable
    csv_fields = [
        'file_name', 'repo_name', 'category', 'title', 'description',
        'file_type', 'file_size', 'creation_date', 'modification_date', 
        'split', 'is_api_spec', 'is_sdk_file'
    ]

    # Create CSV files
    train_csv_path = os.path.join(output_dir, 'train.csv')
    test_csv_path = os.path.join(output_dir, 'test.csv')

    # Write training data to CSV
    with tqdm(desc=f"Creating training CSV ({len(train_data)} records)", total=len(train_data)) as pbar:
        with open(train_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            for item in train_data:
                # Create a copy without the content field
                item_copy = {k: v for k, v in item.items() if k in csv_fields}
                writer.writerow(item_copy)
                pbar.update(1)

    # Write testing data to CSV
    with tqdm(desc=f"Creating testing CSV ({len(test_data)} records)", total=len(test_data)) as pbar:
        with open(test_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            for item in test_data:
                # Create a copy without the content field
                item_copy = {k: v for k, v in item.items() if k in csv_fields}
                writer.writerow(item_copy)
                pbar.update(1)

    # Create a combined metadata CSV
    all_data = train_data + test_data
    metadata_csv_path = os.path.join(output_dir, 'metadata.csv')

    with tqdm(desc=f"Creating metadata CSV ({len(all_data)} records)", total=len(all_data)) as pbar:
        with open(metadata_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            for item in all_data:
                # Create a copy without the content field
                item_copy = {k: v for k, v in item.items() if k in csv_fields}
                writer.writerow(item_copy)
                pbar.update(1)

    tqdm.write(f"Created CSV files in {output_dir}:")
    tqdm.write(f"  - {train_csv_path} ({len(train_data)} records)")
    tqdm.write(f"  - {test_csv_path} ({len(test_data)} records)")
    tqdm.write(f"  - {metadata_csv_path} ({len(all_data)} records)")

def process_repository(repo_info: Dict[str, Any], output_dir: str, temp_dir: str, 
                      github_token: Optional[str] = None, 
                      huggingface_token: Optional[str] = None,
                      test_ratio: float = 0.2) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Process a single repository based on its information.
    
    Args:
        repo_info (Dict[str, Any]): Repository information
        output_dir (str): Output directory
        temp_dir (str): Temporary directory for downloads
        github_token (Optional[str]): GitHub API token
        huggingface_token (Optional[str]): Hugging Face token
        test_ratio (float): Test split ratio
        
    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: Training and testing data
    """
    repo_type = repo_info.get('type', 'repository')
    source = repo_info.get('source', 'github')
    name = repo_info.get('name', '')
    
    all_train_data = []
    all_test_data = []
    
    try:
        if repo_type == 'organization':
            # Process all repositories in the organization
            if source == 'github':
                repos = get_organization_repos_github(name, github_token)
            else:  # huggingface
                repos = get_organization_repos_hf(name, huggingface_token)
                
                # If no repos found on Hugging Face, try GitHub as fallback
                if not repos and source == 'huggingface':
                    tqdm.write(f"No repositories found for {name} on Hugging Face. Trying GitHub...")
                    repos = get_organization_repos_github(name, github_token)
                    source = 'github'  # Update source for subsequent operations

            if not repos:
                tqdm.write(f"No repositories found for organization {name}")
                return [], []
                
            # Process each repository
            for repo_idx, repo in enumerate(repos):
                tqdm.write(f"Processing repository {repo_idx+1}/{len(repos)}")
                
                # Extract repository information based on source
                if source == 'github':
                    if isinstance(repo, dict) and 'clone_url' in repo:
                        repo_url = repo['clone_url']
                        repo_name = repo['name']
                    else:
                        tqdm.write(f"Skipping repository with incomplete information: {repo}")
                        continue

                    # Download GitHub repository
                    repo_path = download_repo_content_github(repo_url, temp_dir, github_token)
                else:
                    # Extract Hugging Face repository ID
                    if hasattr(repo, 'id'):
                        repo_id = repo.id
                    elif isinstance(repo, dict) and 'id' in repo:
                        repo_id = repo['id']
                    else:
                        # Try to extract id from the object
                        repo_id = str(repo)
                        for attr in ['id', 'name', 'modelId']:
                            if hasattr(repo, attr):
                                repo_id = getattr(repo, attr)
                                break

                    # Download Hugging Face repository
                    repo_path = download_repo_content_hf(repo_id, temp_dir, huggingface_token)
                    repo_name = repo_id.split('/')[-1]

                if not repo_path:
                    continue
                    
                # Extract and convert files
                files_by_category = extract_and_convert_files(repo_path, output_dir, repo_name)
                
                # Split into train and test
                train_data, test_data = split_into_train_test(files_by_category, test_ratio)
                all_train_data.extend(train_data)
                all_test_data.extend(test_data)
                
        else:  # Individual repository
            # Extract repository information
            if 'url' in repo_info:
                repo_url = repo_info['url']
                repo_name = repo_info['name'].split('/')[-1]
            else:
                repo_name = repo_info['name']
                org_name = repo_name.split('/')[0] if '/' in repo_name else ''
                repo_short_name = repo_name.split('/')[-1]
                
                if source == 'github':
                    repo_url = f"https://github.com/{org_name}/{repo_short_name}"
                else:  # huggingface
                    repo_url = f"https://huggingface.co/{repo_name}"
            
            # Download repository
            if source == 'github':
                repo_path = download_repo_content_github(repo_url, temp_dir, github_token)
            else:  # huggingface
                repo_path = download_repo_content_hf(repo_name, temp_dir, huggingface_token)
            
            if not repo_path:
                tqdm.write(f"Failed to download repository {repo_name}")
                return [], []
            
            # Extract and convert files
            repo_short_name = repo_name.split('/')[-1] if '/' in repo_name else repo_name
            files_by_category = extract_and_convert_files(repo_path, output_dir, repo_short_name)
            
            # Split into train and test
            train_data, test_data = split_into_train_test(files_by_category, test_ratio)
            all_train_data.extend(train_data)
            all_test_data.extend(test_data)
            
    except Exception as e:
        tqdm.write(f"Error processing repository {name}: {str(e)}")
        logger.error(traceback.format_exc())
    
    return all_train_data, all_test_data

def process_existing_data(output_dir: str, test_ratio: float = 0.2) -> Optional[Tuple[DatasetDict, DatasetDict, DatasetDict]]:
    """
    Process existing downloaded data without re-downloading repositories.
    
    Args:
        output_dir (str): Directory where the extracted content is stored
        test_ratio (float): Ratio of files to use for testing
        
    Returns:
        Optional[Tuple[DatasetDict, DatasetDict, DatasetDict]]: Tuple of (standard dataset, SFT dataset, code generation dataset) or None if processing failed
    """
    try:
        # Check if the directory exists
        if not os.path.exists(output_dir) or not os.path.isdir(output_dir):
            tqdm.write(f"Error: Directory {output_dir} does not exist.")
            return None
        
        # Find all repo directories in the output directory
        repo_dirs = []
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path) and not item in ['__pycache__', 'dataset', 'sft_dataset', 'code_generation_dataset']:
                repo_dirs.append((item, item_path))
        
        if not repo_dirs:
            tqdm.write(f"Error: No repository directories found in {output_dir}.")
            return None
        
        tqdm.write(f"Found {len(repo_dirs)} repository directories to process.")
        
        all_train_data = []
        all_test_data = []
        
        # Define target folders to look for
        target_folders = ['docs', 'examples', 'cookbook', 'cookbooks', 'documentation', 'tutorials', 'other']
        
        with tqdm(total=len(repo_dirs), desc="Processing repositories") as pbar:
            for repo_name, repo_path in repo_dirs:
                pbar.set_description(f"Processing {repo_name}")
                
                files_by_category = {folder: [] for folder in target_folders}
                
                # Find files in each category folder
                for category in target_folders:
                    category_dir = os.path.join(repo_path, category)
                    if os.path.exists(category_dir) and os.path.isdir(category_dir):
                        for root, _, files in os.walk(category_dir):
                            for file in files:
                                file_path = os.path.join(root, file)
                                # Include all files that could be documentation
                                if is_documentation_file(file_path) or file.endswith(('.py', '.ipynb', '.md', '.rst', '.txt')):
                                    files_by_category[category].append(file_path)
                
                # Split into train and test
                train_data, test_data = split_into_train_test(files_by_category, test_ratio)
                all_train_data.extend(train_data)
                all_test_data.extend(test_data)
                
                pbar.update(1)
        
        # Create dataset from collected files
        if all_train_data or all_test_data:
            # Create and standardize the dataset
            dataset = create_dataset_from_files(all_train_data, all_test_data)
            tqdm.write(f"Created dataset with {len(dataset['train'])} training samples and {len(dataset['test'])} testing samples")
            
            # Display column mapping information for SFT
            display_sft_column_mapping(dataset)
            
            # Create SFT-ready dataset
            sft_dataset = prepare_dataset_for_sft(dataset)
            
            # Create code generation specific dataset
            code_gen_dataset = prepare_dataset_for_code_generation(dataset)
            display_code_generation_info(code_gen_dataset)
            
            # Save the datasets locally
            dataset_path = os.path.join(output_dir, "dataset")
            with tqdm(desc=f"Saving standard dataset to {dataset_path}", total=1) as pbar:
                dataset.save_to_disk(dataset_path)
                pbar.update(1)
            
            sft_dataset_path = os.path.join(output_dir, "sft_dataset")
            with tqdm(desc=f"Saving SFT dataset to {sft_dataset_path}", total=1) as pbar:
                sft_dataset.save_to_disk(sft_dataset_path)
                pbar.update(1)
            
            code_gen_path = os.path.join(output_dir, "code_generation_dataset")
            with tqdm(desc=f"Saving code generation dataset to {code_gen_path}", total=1) as pbar:
                code_gen_dataset.save_to_disk(code_gen_path)
                pbar.update(1)
            
            tqdm.write(f"Datasets successfully saved to:")
            tqdm.write(f"  - Standard dataset: {dataset_path}")
            tqdm.write(f"  - SFT-ready dataset: {sft_dataset_path}")
            tqdm.write(f"  - Code Generation dataset: {code_gen_path}")
            
            # Create CSV files as well
            create_csv_files(all_train_data, all_test_data, output_dir)
            
            return dataset, sft_dataset, code_gen_dataset
        else:
            tqdm.write("No documentation files were found for processing")
            return None
    
    except Exception as e:
        tqdm.write(f"Error processing existing data: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def process_all_repositories(config: Dict[str, Any]) -> Optional[Tuple[DatasetDict, DatasetDict, DatasetDict]]:
    """
    Process all repositories in the configuration.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        Optional[Tuple[DatasetDict, DatasetDict, DatasetDict]]: Tuple of (standard dataset, SFT dataset, code generation dataset) or None if processing failed
    """
    repositories = config.get('repositories', [])
    if not repositories:
        tqdm.write("No repositories configured. Please add repositories first.")
        return None
    
    output_dir = config.get('output_directory', './downloaded_docs')
    os.makedirs(output_dir, exist_ok=True)
    
    github_token = config.get('github_token', '')
    huggingface_token = config.get('huggingface_token', '')
    test_ratio = config.get('test_ratio', 0.2)
    
    all_train_data = []
    all_test_data = []
    
    # Create a temporary directory for downloading repositories
    with tempfile.TemporaryDirectory() as temp_dir:
        # Process each repository
        with tqdm(total=len(repositories), desc="Processing repositories") as pbar:
            for i, repo_info in enumerate(repositories):
                repo_name = repo_info.get('name', 'Unknown')
                pbar.set_description(f"Processing {repo_name}")
                
                # Process the repository
                train_data, test_data = process_repository(
                    repo_info, 
                    output_dir, 
                    temp_dir, 
                    github_token, 
                    huggingface_token,
                    test_ratio
                )
                
                all_train_data.extend(train_data)
                all_test_data.extend(test_data)
                pbar.update(1)
    
    # Create CSV files from the collected data
    if all_train_data or all_test_data:
        create_csv_files(all_train_data, all_test_data, output_dir)
        
        # Create and standardize the dataset
        dataset = create_dataset_from_files(all_train_data, all_test_data)
        tqdm.write(f"Created dataset with {len(dataset['train'])} training samples and {len(dataset['test'])} testing samples")
        
        # Display column mapping information for SFT
        display_sft_column_mapping(dataset)
        
        # Create SFT-ready dataset
        sft_dataset = prepare_dataset_for_sft(dataset)
        
        # Create code generation specific dataset
        code_gen_dataset = prepare_dataset_for_code_generation(dataset)
        display_code_generation_info(code_gen_dataset)
        
        # Save the datasets locally
        dataset_path = os.path.join(output_dir, "dataset")
        with tqdm(desc=f"Saving standard dataset to {dataset_path}", total=1) as pbar:
            dataset.save_to_disk(dataset_path)
            pbar.update(1)
        
        sft_dataset_path = os.path.join(output_dir, "sft_dataset")
        with tqdm(desc=f"Saving SFT dataset to {sft_dataset_path}", total=1) as pbar:
            sft_dataset.save_to_disk(sft_dataset_path)
            pbar.update(1)
        
        code_gen_path = os.path.join(output_dir, "code_generation_dataset")
        with tqdm(desc=f"Saving code generation dataset to {code_gen_path}", total=1) as pbar:
            code_gen_dataset.save_to_disk(code_gen_path)
            pbar.update(1)
        
        tqdm.write(f"Datasets successfully saved to:")
        tqdm.write(f"  - Standard dataset: {dataset_path}")
        tqdm.write(f"  - SFT-ready dataset: {sft_dataset_path}")
        tqdm.write(f"  - Code Generation dataset: {code_gen_path}")
        
        return dataset, sft_dataset, code_gen_dataset
    else:
        tqdm.write("No files were found for processing")
        return None

def upload_to_huggingface(dataset: DatasetDict, config: Dict[str, Any], dataset_type: str = "standard") -> bool:
    """
    Upload dataset to Hugging Face Hub.
    
    Args:
        dataset (DatasetDict): Dataset to upload
        config (Dict[str, Any]): Configuration dictionary
        dataset_type (str): Type of dataset ("standard", "sft", or "code_gen")
        
    Returns:
        bool: True if upload was successful, False otherwise
    """
    try:
        # Get default dataset name from the first repository
        default_name = None
        if config['repositories']:
            first_repo = config['repositories'][0]
            repo_name = first_repo.get('name', '').replace('/', '-')
            if repo_name:
                suffix = {
                    "standard": "-docs",
                    "sft": "-docs-sft",
                    "code_gen": "-docs-code-gen"
                }.get(dataset_type, "-docs")
                default_name = f"{repo_name}{suffix}"
        
        if not default_name:
            default_name = {
                "standard": "documentation-dataset",
                "sft": "documentation-dataset-sft",
                "code_gen": "documentation-dataset-code-gen"
            }.get(dataset_type, "documentation-dataset")
        
        # Ask for dataset name
        dataset_name = input_dialog(
            title="Upload to Hugging Face Hub",
            text=f"Enter dataset name for {dataset_type.replace('_', ' ').title()} dataset:",
            default=default_name,
            style=STYLE
        ).run()
        
        if not dataset_name:
            tqdm.write("Upload cancelled.")
            return False
        
        # Confirm upload
        token = config.get('huggingface_token', '')
        if not token:
            message_dialog(
                title="Error",
                text="No Hugging Face token configured. Please add your token first.",
                style=STYLE
            ).run()
            return False
        
        tqdm.write(f"Uploading {dataset_type.replace('_', ' ').title()} dataset to Hugging Face Hub as '{dataset_name}'...")
        
        # Upload to Hugging Face Hub with progress tracking
        with tqdm(desc=f"Uploading to Hugging Face Hub", total=1) as pbar:
            dataset.push_to_hub(
                dataset_name,
                token=token
            )
            pbar.update(1)
        
        tqdm.write(f"Dataset successfully uploaded to Hugging Face Hub as '{dataset_name}'")
        return True
    
    except Exception as e:
        tqdm.write(f"Error uploading dataset to Hugging Face Hub: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def organize_content(organization: str, output_dir: str, use_github: bool = False, test_ratio: float = 0.2) -> Optional[Tuple[DatasetDict, DatasetDict, DatasetDict]]:
    """
    Main function to organize content from an organization's repositories.
    (Maintained for backward compatibility)

    Args:
        organization (str): The name of the organization on Hugging Face Hub or GitHub.
        output_dir (str): Directory where the extracted content will be stored.
        use_github (bool): Whether to use GitHub instead of Hugging Face Hub.
        test_ratio (float): Ratio of files to use for testing (default: 0.2).

    Returns:
        Optional[Tuple[DatasetDict, DatasetDict, DatasetDict]]: Tuple of (standard dataset, SFT dataset, code generation dataset) or None if processing failed
    """
    # Create a temporary config
    config = {
        'repositories': [
            {
                'type': 'organization',
                'name': organization,
                'source': 'github' if use_github else 'huggingface'
            }
        ],
        'output_directory': output_dir,
        'test_ratio': test_ratio,
        'github_token': '',
        'huggingface_token': ''
    }
    
    # Process the repositories
    return process_all_repositories(config)

def main_tui():
    """Main function with TUI interface."""
    # Load or create configuration
    config = load_or_create_config()
    
    # Process command line arguments
    if len(sys.argv) > 1:
        # TODO: Add command line argument handling (if needed)
        pass
    
    # Main program loop
    while True:
        # Count repositories by type
        github_repos = sum(1 for r in config['repositories'] if r.get('source') == 'github')
        hf_repos = sum(1 for r in config['repositories'] if r.get('source') == 'huggingface')
        
        # Main menu
        choice = radiolist_dialog(
            title="SDK Documentation Downloader",
            text="Select an action:",
            values=[
                ("credentials", "Manage Credentials"),
                ("repositories", f"Manage Repositories ({len(config['repositories'])} configured)"),
                ("settings", "Configure Settings"),
                ("start", "Start Download and Processing"),
                ("process_existing", "Process Existing Downloaded Data"),
                ("upload", "Upload Existing Dataset to Hugging Face Hub"),
                ("exit", "Exit")
            ],
            style=STYLE
        ).run()
        
        if choice == "credentials":
            config = manage_credentials(config)
        
        elif choice == "repositories":
            config = manage_repositories(config)
        
        elif choice == "settings":
            config = configure_settings(config)
        
        elif choice == "start":
            if not config['repositories']:
                message_dialog(
                    title="Error",
                    text="No repositories configured. Please add repositories first.",
                    style=STYLE
                ).run()
                continue
            
            # Start processing
            tqdm.write("\nStarting download and processing of repository documentation...")
            result = process_all_repositories(config)
            
            if result:
                dataset, sft_dataset, code_gen_dataset = result
                tqdm.write("\nProcessing completed successfully.")
                
                # Ask if user wants to upload to Hugging Face Hub
                upload_choice = radiolist_dialog(
                    title="Upload Dataset",
                    text="Do you want to upload this dataset to Hugging Face Hub?",
                    values=[
                        ("standard", "Yes, upload standard dataset"),
                        ("sft", "Yes, upload SFT-ready dataset"),
                        ("code_gen", "Yes, upload code generation dataset"),
                        ("all", "Yes, upload all datasets"),
                        ("no", "No, skip upload")
                    ],
                    style=STYLE
                ).run()
                
                if upload_choice == "standard":
                    upload_to_huggingface(dataset, config, "standard")
                elif upload_choice == "sft":
                    upload_to_huggingface(sft_dataset, config, "sft")
                elif upload_choice == "code_gen":
                    upload_to_huggingface(code_gen_dataset, config, "code_gen")
                elif upload_choice == "all":
                    upload_to_huggingface(dataset, config, "standard")
                    upload_to_huggingface(sft_dataset, config, "sft")
                    upload_to_huggingface(code_gen_dataset, config, "code_gen")
            else:
                message_dialog(
                    title="Error",
                    text="Failed to create dataset. Check logs for details.",
                    style=STYLE
                ).run()
        
        elif choice == "process_existing":
            # Ask for the directory with existing data
            dir_path = input_dialog(
                title="Process Existing Data",
                text="Enter the directory path containing the downloaded data:",
                default=config['output_directory'],
                style=STYLE
            ).run()
            
            if dir_path:
                # Ask for test ratio
                test_ratio_str = input_dialog(
                    title="Test Split Ratio",
                    text="Enter the test split ratio (between 0 and 1):",
                    default=str(config['test_ratio']),
                    style=STYLE
                ).run()
                
                test_ratio = config['test_ratio']
                if test_ratio_str:
                    try:
                        test_ratio = float(test_ratio_str)
                        if not 0 <= test_ratio <= 1:
                            test_ratio = config['test_ratio']
                            tqdm.write(f"Invalid test ratio. Using default: {test_ratio}")
                    except ValueError:
                        tqdm.write(f"Invalid test ratio. Using default: {test_ratio}")
                
                # Process the existing data
                tqdm.write(f"\nProcessing existing data in {dir_path} with test ratio {test_ratio}...")
                result = process_existing_data(dir_path, test_ratio)
                
                if result:
                    dataset, sft_dataset, code_gen_dataset = result
                    tqdm.write("\nProcessing of existing data completed successfully.")
                    
                    # Ask if user wants to upload to Hugging Face Hub
                    upload_choice = radiolist_dialog(
                        title="Upload Dataset",
                        text="Do you want to upload this dataset to Hugging Face Hub?",
                        values=[
                            ("standard", "Yes, upload standard dataset"),
                            ("sft", "Yes, upload SFT-ready dataset"),
                            ("code_gen", "Yes, upload code generation dataset"),
                            ("all", "Yes, upload all datasets"),
                            ("no", "No, skip upload")
                        ],
                        style=STYLE
                    ).run()
                    
                    if upload_choice == "standard":
                        upload_to_huggingface(dataset, config, "standard")
                    elif upload_choice == "sft":
                        upload_to_huggingface(sft_dataset, config, "sft")
                    elif upload_choice == "code_gen":
                        upload_to_huggingface(code_gen_dataset, config, "code_gen")
                    elif upload_choice == "all":
                        upload_to_huggingface(dataset, config, "standard")
                        upload_to_huggingface(sft_dataset, config, "sft")
                        upload_to_huggingface(code_gen_dataset, config, "code_gen")
                else:
                    message_dialog(
                        title="Error",
                        text="Failed to process existing data. Check logs for details.",
                        style=STYLE
                    ).run()
        
        elif choice == "upload":
            # Ask which type of dataset to upload
            dataset_type = radiolist_dialog(
                title="Upload Dataset Type",
                text="Which type of dataset do you want to upload?",
                values=[
                    ("standard", "Standard Dataset"),
                    ("sft", "SFT-Ready Dataset"),
                    ("code_gen", "Code Generation Dataset")
                ],
                style=STYLE
            ).run()
            
            if dataset_type:
                # Determine path based on dataset type
                path_suffix = {
                    "standard": "dataset",
                    "sft": "sft_dataset",
                    "code_gen": "code_generation_dataset"
                }.get(dataset_type, "dataset")
                
                default_path = os.path.join(config['output_directory'], path_suffix)
                
                dataset_path = input_dialog(
                    title="Upload Existing Dataset",
                    text=f"Enter path to the {dataset_type.replace('_', ' ').title()} dataset:",
                    default=default_path,
                    style=STYLE
                ).run()
                
                if dataset_path:
                    try:
                        # Load the dataset
                        tqdm.write(f"Loading dataset from {dataset_path}...")
                        with tqdm(desc="Loading dataset", total=1) as pbar:
                            dataset = DatasetDict.load_from_disk(dataset_path)
                            pbar.update(1)
                        
                        tqdm.write(f"Dataset loaded with {len(dataset['train'])} training samples")
                        if 'test' in dataset:
                            tqdm.write(f"and {len(dataset['test'])} testing samples")
                        
                        # Upload the dataset
                        upload_to_huggingface(dataset, config, dataset_type)
                    except Exception as e:
                        message_dialog(
                            title="Error",
                            text=f"Error loading dataset: {str(e)}",
                            style=STYLE
                        ).run()
        
        elif choice == "exit" or choice is None:
            tqdm.write("Exiting...")
            break

def main():
    """Legacy main function (kept for backward compatibility)."""
    message_dialog(
        title="Information",
        text="This script now uses a text-based user interface (TUI). Switching to TUI mode...",
        style=STYLE
    ).run()
    
    main_tui()

if __name__ == "__main__":
    main_tui()
