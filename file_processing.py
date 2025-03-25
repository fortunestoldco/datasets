#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import shutil
import json
import logging
import traceback
from typing import Dict, Any, Optional, Tuple

from utils import is_documentation_file, get_file_metadata, standardize_metadata
from converters import (
    convert_ipynb_to_md, convert_py_to_md_enhanced,
    process_api_specification
)

logger = logging.getLogger(__name__)

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
                    try:
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
                    except Exception:
                        # If we can't parse as JSON, skip it
                        pass
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
        logger.debug(traceback.format_exc())

    return None

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
        logger.debug(traceback.format_exc())
        return None