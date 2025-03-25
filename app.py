import os
import re
import csv
import json
import time
import tempfile
import shutil
import base64
import zipfile
import requests
import gradio as gr
import nbformat
from pathlib import Path
from github import Github, GithubException
from bs4 import BeautifulSoup
from markdown import markdown
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GithubOrgDatasetGenerator:
    def __init__(self):
        self.temp_dir = None
        self.dataset_dir = None
        self.train_dir = None
        self.test_dir = None
        self.metadata = {
            "files": [],
            "modules": set(),
            "file_counts": {"py": 0, "ipynb": 0, "md": 0, "rst": 0, "txt": 0, "html": 0, "other": 0}
        }
        
    def setup_directories(self):
        """Set up temporary and dataset directories"""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_dir = os.path.join(self.temp_dir, "dataset")
        self.train_dir = os.path.join(self.dataset_dir, "train")
        self.test_dir = os.path.join(self.dataset_dir, "test")
        
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.test_dir, exist_ok=True)
        
    def cleanup(self):
        """Clean up temporary directories"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            
    def extract_org_name(self, url):
        """Extract organization name from GitHub URL"""
        pattern = r'github\.com/([^/]+)'
        match = re.search(pattern, url)
        if match:
            return match.group(1)
        return None
    
    def extract_text_from_markdown(self, content):
        """Convert markdown to plain text"""
        html = markdown(content)
        soup = BeautifulSoup(html, features="html.parser")
        return soup.get_text()
    
    def extract_text_from_notebook(self, content):
        """Extract text from Jupyter notebook"""
        try:
            notebook = nbformat.reads(content, as_version=4)
            text_content = []
            
            for cell in notebook.cells:
                if cell.cell_type == "markdown":
                    text_content.append(self.extract_text_from_markdown(cell.source))
                elif cell.cell_type == "code":
                    text_content.append(f"```python\n{cell.source}\n```")
            
            return "\n\n".join(text_content)
        except Exception as e:
            logging.error(f"Error parsing notebook: {str(e)}")
            return f"Error parsing notebook: {str(e)}"
    
    def process_file(self, repo, file_path, target_dir, module_name, category):
        """Process and save a file"""
        try:
            logging.info(f"Processing file: {file_path}")
            file_content = repo.get_contents(file_path).decoded_content.decode('utf-8', errors='replace')
            
            if not file_content:
                logging.error(f"File content is empty: {file_path}")
                return False
            
            # Extract file extension and name
            _, ext = os.path.splitext(file_path)
            file_name = os.path.basename(file_path)
            ext = ext.lower().lstrip('.')
            
            # Update file count
            if ext in ['py', 'ipynb', 'md', 'rst', 'txt', 'html']:
                self.metadata["file_counts"][ext] += 1
            else:
                self.metadata["file_counts"]["other"] += 1
            
            # Process content based on file type
            if ext == 'md':
                plain_text = self.extract_text_from_markdown(file_content)
            elif ext == 'ipynb':
                plain_text = self.extract_text_from_notebook(file_content)
            else:
                plain_text = file_content
            
            # Create directory structure
            save_dir = os.path.join(target_dir, module_name, category)
            os.makedirs(save_dir, exist_ok=True)
            
            # Save file
            save_path = os.path.join(save_dir, file_name)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(plain_text)
            
            # Get title from content
            title = file_name
            if ext == 'md':
                # Try to extract title from markdown
                lines = file_content.strip().split('\n')
                for line in lines:
                    if line.startswith('# '):
                        title = line.replace('# ', '').strip()
                        break
            
            # Extract description
            description = plain_text[:300] + '...' if len(plain_text) > 300 else plain_text
            
            # Add to metadata
            self.metadata["files"].append({
                "file_name": file_name,
                "module": module_name,
                "category": category,
                "title": title,
                "description": description,
                "path": os.path.join(module_name, category, file_name),
                "extension": ext
            })
            
            return True
        except FileNotFoundError as e:
            logging.error(f"File not found: {file_path} - {str(e)}")
            return False
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {str(e)}")
            return False
    
    def process_directory(self, repo, dir_path, target_dir, module_name, category):
        """Process a directory recursively"""
        try:
            contents = repo.get_contents(dir_path)
            
            if not contents:
                return False
            
            for content in contents:
                if content.type == "dir":
                    # Create subdirectory with same category
                    self.process_directory(repo, content.path, target_dir, module_name, category)
                elif content.type == "file":
                    # Only process files with specific extensions
                    _, ext = os.path.splitext(content.name)
                    ext = ext.lower().lstrip('.')
                    if ext in ['py', 'ipynb', 'md', 'rst', 'txt', 'html']:
                        self.process_file(repo, content.path, target_dir, module_name, category)
            
            return True
        except Exception as e:
            logging.error(f"Error processing directory {dir_path}: {str(e)}")
            return False
    
    def create_csv_files(self):
        """Create CSV files for the dataset"""
        # Create train.csv
        train_csv_path = os.path.join(self.dataset_dir, "train.csv")
        test_csv_path = os.path.join(self.dataset_dir, "test.csv")
        
        train_data = []
        test_data = []
        
        if not self.metadata["files"]:
            return False
        
        # Split data: 80% train, 20% test
        for i, file_info in enumerate(self.metadata["files"]):
            if i % 5 == 0:  # 20% for test
                test_data.append(file_info)
            else:
                train_data.append(file_info)
        
        # Define CSV columns
        columns = ["file_name", "module", "category", "title", "description", "path", "extension"]
        
        # Write train CSV
        with open(train_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for data in train_data:
                writer.writerow(data)
        
        # Write test CSV
        with open(test_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for data in test_data:
                writer.writerow(data)
        
        # Create dataset_info.json - Fixed the attribute name from "id" to "_id"
        dataset_info = {
            "description": "Dataset created from GitHub organization documentation and examples",
            "citation": "",
            "homepage": "",
            "license": "",
            "features": {
                "file_name": {"dtype": "string", "_id": None, "_type": "Value"},
                "module": {"dtype": "string", "_id": None, "_type": "Value"},
                "category": {"dtype": "string", "_id": None, "_type": "Value"},
                "title": {"dtype": "string", "_id": None, "_type": "Value"},
                "description": {"dtype": "string", "_id": None, "_type": "Value"},
                "path": {"dtype": "string", "_id": None, "_type": "Value"},
                "extension": {"dtype": "string", "_id": None, "_type": "Value"}
            },
            "splits": {
                "train": {"name": "train", "num_bytes": 0, "num_examples": len(train_data), "dataset_name": "github_docs"},
                "test": {"name": "test", "num_bytes": 0, "num_examples": len(test_data), "dataset_name": "github_docs"}
            },
            "download_size": 0,
            "dataset_size": 0
        }
        
        with open(os.path.join(self.dataset_dir, "dataset_info.json"), 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2)
            
        # Create README.md
        readme_content = f"""# GitHub Documentation Dataset

This dataset contains documentation, examples, and cookbooks from GitHub repositories.

## Dataset Structure

- Total files: {sum(self.metadata["file_counts"].values())}
  - Python files (.py): {self.metadata["file_counts"]["py"]}
  - Jupyter notebooks (.ipynb): {self.metadata["file_counts"]["ipynb"]}
  - Markdown files (.md): {self.metadata["file_counts"]["md"]}
  - reStructuredText files (.rst): {self.metadata["file_counts"]["rst"]}
  - Text files (.txt): {self.metadata["file_counts"]["txt"]}
  - HTML files (.html): {self.metadata["file_counts"]["html"]}
  - Other files: {self.metadata["file_counts"]["other"]}

## Modules

{', '.join(self.metadata["modules"])}

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("path/to/dataset")
```
"""
        
        with open(os.path.join(self.dataset_dir, "README.md"), 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
    def process_organization(self, org_url, github_token=None, max_repos=None, progress=None, include_private=False):
        """Process all repositories in a GitHub organization"""
        try:
            # Extract organization name
            org_name = self.extract_org_name(org_url)
            if not org_name:
                return False, "Invalid GitHub organization URL"
            
            # Setup directories
            self.setup_directories()
            
            # Initialize GitHub client
            g = Github(github_token) if github_token else Github()
            
            try:
                # Get organization
                org = g.get_organization(org_name)
                
                # Get repositories
                try:
                    # Use pagination to avoid memory issues and potential list index errors
                    repos = []
                    for repo in org.get_repos(type='all' if include_private else 'public'):
                        repos.append(repo)
                        
                    # Check if there are any repositories
                    if not repos:
                        return False, "No repositories found in the organization"
                    
                    if max_repos:
                        try:
                            max_repos_int = int(max_repos)
                            repos = repos[:max_repos_int]
                        except (ValueError, TypeError):
                            return False, "Invalid 'max_repos' value, must be a positive integer"
                    
                    total_repos = len(repos)
                    
                    # Process each repository
                    with ThreadPoolExecutor() as executor:
                        futures = []
                        for i, repo in enumerate(repos):
                            futures.append(executor.submit(self.process_repository, repo, i, total_repos, progress))
                        
                        # Improved error handling for futures
                        for future in as_completed(futures):
                            try:
                                future.result()
                            except Exception as e:
                                logging.error(f"Error in worker thread: {str(e)}")
                                # Don't immediately fail, continue processing other repos
                except Exception as e:
                    return False, f"Error listing repositories: {str(e)}"
                
                # Create CSV files and other metadata
                if progress:
                    try:
                        progress(0.9, "Creating dataset files...")
                    except Exception as e:
                        logging.warning(f"Error updating progress: {str(e)}")
                
                self.create_csv_files()
                
                # Create zip file
                if progress:
                    try:
                        progress(0.95, "Creating zip archive...")
                    except Exception as e:
                        logging.warning(f"Error updating progress: {str(e)}")
                
                zip_path = os.path.join(self.temp_dir, "dataset.zip")
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, _, files in os.walk(self.dataset_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, self.temp_dir)
                            zipf.write(file_path, arcname)
                
                if progress:
                    try:
                        progress(1.0, "Dataset generation complete!")
                    except Exception as e:
                        logging.warning(f"Error updating progress: {str(e)}")
                
                return True, zip_path
                
            except GithubException as e:
                return False, f"GitHub API error: {str(e)}"
            
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}", exc_info=True)
            return False, f"An unexpected error occurred: {str(e)}"
        finally:
            # Clean up temporary files, but keep the zip
            pass

    def process_repository(self, repo, index, total_repos, progress):
        """Process a single repository"""
        if progress:
            try:
                progress(index/total_repos, f"Processing repository {index+1}/{total_repos}: {repo.name}")
            except Exception as e:
                logging.warning(f"Error updating progress: {str(e)}")
        
        module_name = repo.name
        self.metadata["modules"].add(module_name)
        
        # Look for documentation directories
        try:
            # Check for docs directory
            try:
                docs_contents = repo.get_contents("docs")
                if docs_contents:
                    self.process_directory(repo, "docs", self.train_dir, module_name, "docs")
            except Exception:
                pass
            
            # Check for doc directory
            try:
                doc_contents = repo.get_contents("doc")
                if doc_contents:
                    self.process_directory(repo, "doc", self.train_dir, module_name, "docs")
            except Exception:
                pass
            
            # Check for examples directory
            try:
                examples_contents = repo.get_contents("examples")
                if examples_contents:
                    self.process_directory(repo, "examples", self.train_dir, module_name, "examples")
            except Exception:
                pass
                
            # Check for example directory
            try:
                example_contents = repo.get_contents("example")
                if example_contents:
                    self.process_directory(repo, "example", self.train_dir, module_name, "examples")
            except Exception:
                pass
            
            # Check for cookbooks directory
            try:
                cookbooks_contents = repo.get_contents("cookbooks")
                if cookbooks_contents:
                    self.process_directory(repo, "cookbooks", self.train_dir, module_name, "cookbooks")
            except Exception:
                pass
                
            # Check for cookbook directory
            try:
                cookbook_contents = repo.get_contents("cookbook")
                if cookbook_contents:
                    self.process_directory(repo, "cookbook", self.train_dir, module_name, "cookbooks")
            except Exception:
                pass
                
            # Check for tutorials directory
            try:
                tutorials_contents = repo.get_contents("tutorials")
                if tutorials_contents:
                    self.process_directory(repo, "tutorials", self.train_dir, module_name, "examples")
            except Exception:
                pass
        
        except Exception as e:
            if "list index out of range" in str(e):
                logging.error(f"List index out of range error in repository {repo.name}: {str(e)}")
            else:
                logging.error(f"Error processing repository {repo.name}: {str(e)}")

def generate_dataset(org_url, github_token, max_repos, progress=gr.Progress(), include_private=False):
    """Generate dataset from GitHub organization"""
    generator = GithubOrgDatasetGenerator()
    success, result = generator.process_organization(org_url, github_token, max_repos, progress, include_private)
    
    if success:
        # Return the path to the zip file
        return result
    else:
        # Always raise errors rather than returning strings
        raise gr.Error(result)

def validate_github_url(url):
    """Validate that the input is a GitHub org URL"""
    if not url:
        return False, "Please enter a GitHub organization URL"
    
    if not url.startswith("https://github.com/") and not url.startswith("http://github.com/"):
        return False, "URL must be a GitHub URL (https://github.com/organization)"
    
    # Check if it's an organization URL by making a request
    try:
        # Extract org name
        pattern = r'github\.com/([^/]+)'
        match = re.search(pattern, url)
        if not match:
            return False, "Invalid GitHub URL format"
        
        org_name = match.group(1)
        response = requests.get(f"https://github.com/{org_name}")
        
        if response.status_code != 200:
            return False, f"Organization '{org_name}' not found"
        
        # Check if it's an organization page by looking for organization-specific elements
        soup = BeautifulSoup(response.text, 'html.parser')
        org_indicators = soup.select('meta[name="hovercard-subject-tag"][content="organization"]')
        
        if not org_indicators:
            return False, f"'{org_name}' appears to be a user, not an organization"
        
        return True, ""
    except Exception as e:
        return False, f"Error validating URL: {str(e)}"

def create_sample_button_click():
    """Function to handle the 'Use Sample Org' button click"""
    return "https://github.com/huggingface"

def download_file(file_path):
    """Function to handle file download"""
    return file_path

def clean_temp_files():
    """Function to clean temporary files"""
    temp_dirs = [d for d in os.listdir('/tmp') if d.startswith('tmp')]
    for d in temp_dirs:
        try:
            path = os.path.join('/tmp', d)
            if os.path.isdir(path) and (time.time() - os.path.getmtime(path)) > 3600:  # Older than 1 hour
                shutil.rmtree(path)
        except Exception:
            pass

# Create the Gradio interface
with gr.Blocks(title="GitHub Organization Dataset Generator") as app:
    gr.Markdown("""
    # GitHub Organization Dataset Generator
    
    This tool creates a HuggingFace-compatible dataset from documentation and examples in a GitHub organization's repositories.
    
    ## Instructions:
    1. Enter a GitHub organization URL (e.g., https://github.com/huggingface)
    2. Optionally provide a GitHub token for API rate limit increase
    3. Click 'Generate Dataset' to start the process
    4. Download the generated dataset zip file
    
    The tool will extract content from the following directories in each repository:
    - docs / doc (documentation)
    - examples / example / tutorials (code examples)
    - cookbooks / cookbook (recipes and guides)
    """)
    
    with gr.Row():
        with gr.Column():
            org_url = gr.Textbox(
                label="GitHub Organization URL",
                placeholder="https://github.com/organization",
                info="URL of the GitHub organization to process"
            )
            
            github_token = gr.Textbox(
                label="GitHub API Token (Optional)",
                placeholder="ghp_xxxxxxxxxxxx",
                info="Provide a token to increase API rate limits",
                type="password"
            )
            
            max_repos = gr.Number(
                label="Maximum Repositories (Optional)",
                value=10,
                info="Limit the number of repositories to process (leave empty for all)",
                minimum=1,
                step=1
            )
            
            include_private = gr.Checkbox(
                label="Include Private Repositories",
                info="Check this box to include private repositories (requires appropriate GitHub token permissions)"
            )
            
            with gr.Row():
                sample_button = gr.Button("Use Sample Org (HuggingFace)")
                generate_button = gr.Button("Generate Dataset", variant="primary")
        
        with gr.Column():
            output = gr.File(label="Generated Dataset")
            
    # Set up event handlers
    sample_button.click(
        fn=create_sample_button_click,
        outputs=org_url
    )
    
    generate_button.click(
        fn=generate_dataset,
        inputs=[org_url, github_token, max_repos, include_private],
        outputs=output
    )
    
    # Clean temporary files on load
    clean_temp_files()

if __name__ == "__main__":
    app.launch(share=True)
