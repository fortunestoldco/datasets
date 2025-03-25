#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import csv
import random
import logging
import traceback
import multiprocessing
import re
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from datasets import Dataset, DatasetDict
except ImportError:
    # These will be handled elsewhere in the dynamic import code
    pass

from file_processing import process_file_for_dataset
from utils import standardize_metadata

logger = logging.getLogger(__name__)

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
    tqdm.write(r"""
    # Define a formatting function to combine title and content
    def formatting_func(example):
        text = f"# {example['title']}\n\n{example['content']}"
        return {"text": text}
    """)

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
                function_calls.extend(re.findall(r'([a-zA-Z0-9_]+\\(.*?\\))', ex))

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

def upload_to_huggingface(dataset: DatasetDict, config: Dict[str, Any], dataset_type: str = "standard", dataset_name: str = None) -> bool:
    """
    Upload dataset to Hugging Face Hub.

    Args:
        dataset (DatasetDict): Dataset to upload
        config (Dict[str, Any]): Configuration dictionary
        dataset_type (str): Type of dataset ("standard", "sft", or "code_gen")
        dataset_name (str): Optional name for the dataset (if None, a default is used)

    Returns:
        bool: True if upload was successful, False otherwise
    """
    try:
        # Get default dataset name from the first repository if not provided
        if dataset_name is None:
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

            # Use default name
            dataset_name = default_name

        if not dataset_name:
            tqdm.write("Upload cancelled.")
            return False

        # Confirm upload
        token = config.get('huggingface_token', '')
        if not token:
            tqdm.write("Error: No Hugging Face token configured. Please add your token first.")
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