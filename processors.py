#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import tempfile
import logging
import traceback
from typing import Dict, Any, Optional, Tuple, List
from tqdm import tqdm

try:
    from datasets import DatasetDict
except ImportError:
    # Will be handled elsewhere in dynamic import
    pass

from repo_utils import (
    get_organization_repos_github,
    download_repo_content_github,
    extract_and_convert_files
)
from dataset_utils import (
    split_into_train_test,
    create_dataset_from_files,
    create_csv_files,
    prepare_dataset_for_sft,
    prepare_dataset_for_code_generation,
    display_sft_column_mapping,
    display_code_generation_info
)
from utils import is_documentation_file

logger = logging.getLogger(__name__)

def process_repository(repo_info: Dict[str, Any], output_dir: str, temp_dir: str,
                      github_token: Optional[str] = None,
                      huggingface_token: Optional[str] = None,
                      test_ratio: float = 0.2) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Process a single repository based on its information.
    """
    repo_type = repo_info.get('type', 'repository')
    # Always use GitHub as source
    name = repo_info.get('name', '')

    all_train_data = []
    all_test_data = []

    try:
        if repo_type == 'organization':
            # Process all repositories in the organization
            repos = get_organization_repos_github(name, github_token)

            if not repos:
                tqdm.write(f"No repositories found for organization {name}")
                return [], []

            # Process each repository
            for repo_idx, repo in enumerate(repos):
                tqdm.write(f"Processing repository {repo_idx+1}/{len(repos)}")

                # Extract repository information
                if isinstance(repo, dict) and 'clone_url' in repo:
                    repo_url = repo['clone_url']
                    repo_name = repo['name']
                else:
                    tqdm.write(f"Skipping repository with incomplete information: {repo}")
                    continue

                # Download GitHub repository
                repo_path = download_repo_content_github(repo_url, temp_dir, github_token)

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
                repo_name = repo_info['name']
            else:
                repo_name = repo_info['name']
                org_name = repo_name.split('/')[0] if '/' in repo_name else ''
                repo_short_name = repo_name.split('/')[-1]

                repo_url = f"https://github.com/{org_name}/{repo_short_name}"

            # Download repository
            repo_path = download_repo_content_github(repo_url, temp_dir, github_token)

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

def organize_content(organization: str, output_dir: str, test_ratio: float = 0.2) -> Optional[Tuple[DatasetDict, DatasetDict, DatasetDict]]:
    """
    Main function to organize content from an organization's repositories.
    (Maintained for backward compatibility)
    """
    # Create a temporary config
    config = {
        'repositories': [
            {
                'type': 'organization',
                'name': organization,
                'source': 'github'  # Always use GitHub
            }
        ],
        'output_directory': output_dir,
        'test_ratio': test_ratio,
        'github_token': '',
        'huggingface_token': ''
    }

    # Process the repositories
    return process_all_repositories(config)
