#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import json
import datetime
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

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
