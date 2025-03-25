#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import logging
import traceback
import requests
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import git
    from huggingface_hub import HfApi, snapshot_download
except ImportError:
    # These will be handled elsewhere in the dynamic import code
    pass

from file_processing import process_file

logger = logging.getLogger(__name__)

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

def get_organization_repos_github(organization: str, token: Optional[str] = None, 
                                retry_limit: int = 3, retry_delay: int = 60) -> List[Dict[str, Any]]:
    """
    Retrieve all repositories for a given organization from GitHub.

    Args:
        organization (str): The name of the organization on GitHub.
        token (Optional[str]): GitHub API token.
        retry_limit (int): Number of retries for rate-limited requests.
        retry_delay (int): Delay in seconds between retries.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing repository information.
    """
    try:
        url = f"https://api.github.com/orgs/{organization}/repos?per_page=100"
        headers = {"Authorization": f"token {token}"} if token else {}
        repos = []
        page = 1
        retries = 0

        with tqdm(desc=f"Fetching GitHub repositories for {organization}", unit="page") as pbar:
            while True:
                try:
                    response = requests.get(f"{url}&page={page}", headers=headers)

                    if response.status_code == 403 and 'rate limit exceeded' in response.text.lower():
                        if retries >= retry_limit:
                            tqdm.write("GitHub API rate limit exceeded and retry limit reached.")
                            break
                        
                        # Get reset time from headers
                        reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
                        wait_time = max(reset_time - time.time(), retry_delay)
                        
                        tqdm.write(f"Rate limit exceeded. Waiting {wait_time:.0f} seconds...")
                        time.sleep(wait_time)
                        retries += 1
                        continue

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

                    if 'next' not in response.links:
                        break

                except requests.RequestException as e:
                    if retries >= retry_limit:
                        tqdm.write(f"Error fetching repositories: {str(e)}")
                        break
                    time.sleep(retry_delay)
                    retries += 1
                    continue

        tqdm.write(f"Found {len(repos)} repositories for {organization} on GitHub.")
        return repos

    except Exception as e:
        tqdm.write(f"Error retrieving repositories for {organization} from GitHub: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def check_repo_for_folders(repo_url: str, token: Optional[str] = None) -> bool:
    """
    Check if the repository contains documentation, examples, or cookbooks folders.

    Args:
        repo_url (str): The URL of the repository.
        token (Optional[str]): API token for authentication.

    Returns:
        bool: True if any of the folders are found, False otherwise.
    """
    try:
        headers = {"Authorization": f"token {token}"} if token else {}
        response = requests.get(repo_url, headers=headers)

        if response.status_code != 200:
            tqdm.write(f"Failed to access repository {repo_url}. Status code: {response.status_code}")
            return False

        repo_content = response.json()
        folder_names = [item['name'].lower() for item in repo_content if item['type'] == 'dir']

        target_folders = ['docs', 'examples', 'cookbook', 'cookbooks', 'documentation', 'tutorials']
        for folder in target_folders:
            if folder in folder_names:
                return True

        return False
    except Exception as e:
        tqdm.write(f"Error checking repository {repo_url} for folders: {str(e)}")
        return False

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
        repo_url = f"https://huggingface.co/api/models/{repo_id}/tree/main"
        if not check_repo_for_folders(repo_url, token):
            tqdm.write(f"Skipping download of {repo_id} as it does not contain documentation, examples, or cookbooks.")
            return None

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
        if not check_repo_for_folders(repo_url, token):
            tqdm.write(f"Skipping download of {repo_url} as it does not contain documentation, examples, or cookbooks.")
            return None

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
