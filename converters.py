#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import json
import logging
import warnings
import traceback
import nbformat
from nbconvert import MarkdownExporter

logger = logging.getLogger(__name__)

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

def extract_function_signature(py_path: str) -> list:
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

                    if (char == ',' and bracket_depth == 0) or char == ')':
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