#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
import time  # Add for rate limit handling
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Any, Optional, Set, Union
import logging
from pathlib import Path
import argparse
import dotenv  # Add for environment variable support

# Load environment variables
dotenv.load_dotenv()

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
    "gitpython", "tqdm", "textual"
]

try:
    from huggingface_hub import HfApi, Repository, hf_hub_download, snapshot_download
    from datasets import Dataset, DatasetDict
    import nbformat
    from nbconvert import MarkdownExporter
    import git
    from tqdm import tqdm
    from tqdm.contrib.concurrent import process_map
    from textual import on, work
    from textual.app import App, ComposeResult
    from textual.widgets import Button, Input, Label, ListView, ListItem, Header, Footer
    from textual.scroll_view import ScrollView
    from textual.reactive import reactive
    from textual import events
    from textual.widget import Widget
    from textual.containers import Container, Grid, Horizontal, Vertical
    from textual.screen import ModalScreen
    try:
        from textual.widgets import MessageDialog
    except ImportError:
        # Define a fallback if needed
        pass
except ImportError as e:
    missing_lib = str(e).split("'")[1]
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
    from textual import on, work
    from textual.app import App, ComposeResult
    from textual.widgets import Button, Input, Label, ListView, ListItem, Header, Footer
    from textual.scroll_view import ScrollView
    from textual.reactive import reactive
    from textual import events
    from textual.widget import Widget
    from textual.containers import Container, Grid, Horizontal, Vertical
    from textual.screen import ModalScreen

# Configuration file path
CONFIG_FILE = os.path.expanduser("~/.doc_downloader_config.json")

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
    default_config = {
        'github_token': '',
        'huggingface_token': '',  # Keep this for uploads
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

# Custom dialog screens for TUI
class MessageDialog(ModalScreen):
    """A simple dialog that shows a message and an OK button."""

    DEFAULT_CSS = """
    MessageDialog {
        align: center middle;
    }

    #dialog {
        width: 60;
        height: auto;
        padding: 1 2;
        background: $surface;
        border: thick $primary;
        margin: 1;
    }

    #title {
        width: 100%;
        margin-bottom: 1;
        text-style: bold;
        text-align: center;
    }

    #message {
        width: 100%;
        margin-bottom: 1;
        text-align: center;
    }

    #buttons {
        width: 100%;
        height: 3;
        align: center middle;
        margin-top: 1;
    }
    """

    def __init__(self, title: str, message: str) -> None:
        super().__init__()
        self.dialog_title = title
        self.dialog_message = message

    def compose(self) -> ComposeResult:
        with Container(id="dialog"):
            yield Label(self.dialog_title, id="title")
            yield Label(self.dialog_message, id="message")
            with Horizontal(id="buttons"):
                yield Button("OK", variant="primary", id="ok")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss()

class InputDialog(ModalScreen[str]):
    """A modal dialog for text input."""

    DEFAULT_CSS = """
    InputDialog {
        align: center middle;
    }

    #dialog {
        width: 60;
        height: auto;
        padding: 1 2;
        background: $surface;
        border: thick $primary;
        margin: 1;
    }

    #question {
        text-align: center;
        margin-bottom: 1;
    }

    #input {
        margin-bottom: 1;
    }

    #buttons {
        width: 100%;
        height: 3;
        align: center middle;
        margin-top: 1;
    }

    Button {
        margin: 0 1;
    }
    """

    def __init__(self, question: str, default_value: str = "") -> None:
        super().__init__()
        self.question = question
        self.default_value = default_value

    def compose(self) -> ComposeResult:
        with Container(id="dialog"):
            yield Label(self.question, id="question")
            yield Input(value=self.default_value, id="input")
            with Horizontal(id="buttons"):
                yield Button("OK", variant="primary", id="ok")
                yield Button("Cancel", variant="error", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ok":
            self.dismiss(self.query_one("#input", Input).value)
        else:
            self.dismiss("")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)


class SelectDialog(ModalScreen[str]):
    """A modal dialog for selecting options."""

    DEFAULT_CSS = """
    SelectDialog {
        align: center middle;
    }

    #dialog {
        width: 60;
        height: auto;
        padding: 1 2;
        background: $surface;
        border: thick $primary;
        margin: 1;
    }

    #title {
        text-align: center;
        margin-bottom: 1;
        text-style: bold;
    }

    #subtitle {
        text-align: center;
        margin-bottom: 1;
    }

    #options {
        height: auto;
        max-height: 10;
        margin: 1;
        border: solid $panel;
    }

    #buttons {
        width: 100%;
        height: 3;
        align: center middle;
        margin-top: 1;
    }

    Button {
        margin: 0 1;
    }
    """

    def __init__(self, title: str, subtitle: str, options: List[Tuple[str, str]]) -> None:
        super().__init__()
        self.title_text = title
        self.subtitle_text = subtitle
        self.options = options

    def compose(self) -> ComposeResult:
        with Container(id="dialog"):
            yield Label(self.title_text, id="title")
            yield Label(self.subtitle_text, id="subtitle")
            yield ListView(
                *[ListItem(Label(option[1]), id=option[0]) for option in self.options],
                id="options"
            )
            with Horizontal(id="buttons"):
                yield Button("Cancel", variant="error", id="cancel")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.dismiss(event.item.id)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss("")


# TUI Screens (converted from App to ModalScreen)
class ManageCredentialsScreen(ModalScreen[Dict[str, Any]]):
    """Screen for managing credentials."""

    DEFAULT_CSS = """
    ManageCredentialsScreen {
        align: center middle;
    }

    #credentials-container {
        width: 60;
        height: auto;
        padding: 1 2;
        background: $surface;
        border: thick $primary;
        margin: 1;
    }

    #title {
        text-align: center;
        margin-bottom: 1;
        text-style: bold;
    }

    Label {
        margin-bottom: 1;
    }

    Input {
        margin-bottom: 1;
    }

    #buttons {
        width: 100%;
        height: 3;
        align: center middle;
        margin-top: 1;
    }

    Button {
        margin: 0 1;
    }
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config.copy()

    def compose(self) -> ComposeResult:
        with Container(id="credentials-container"):
            yield Label("Manage Credentials", id="title", classes="title")
            yield Label("GitHub Token:")
            yield Input(value=self.config['github_token'], placeholder="Enter your GitHub token", id="github-token")
            yield Label("Hugging Face Token:")
            yield Input(value=self.config['huggingface_token'], placeholder="Enter your Hugging Face token", id="huggingface-token")
            with Horizontal(id="buttons"):
                yield Button("Save", id="save", variant="success")
                yield Button("Cancel", variant="error", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save":
            self.config['github_token'] = self.query_one("#github-token", Input).value
            self.config['huggingface_token'] = self.query_one("#huggingface-token", Input).value
            save_config(self.config)
            self.dismiss(self.config)
        else:
            self.dismiss(None)


class ManageRepositoriesScreen(ModalScreen[Dict[str, Any]]):
    """Screen for managing repositories."""

    DEFAULT_CSS = """
    ManageRepositoriesScreen {
        align: center middle;
    }

    #repositories-container {
        width: 60;
        height: auto;
        padding: 1 2;
        background: $surface;
        border: thick $primary;
        margin: 1;
    }

    #title {
        text-align: center;
        margin-bottom: 1;
        text-style: bold;
    }

    Label {
        text-align: center;
        margin-bottom: 1;
    }

    #repo-list {
        height: 10;
        border: solid $accent;
        margin: 1;
    }

    #buttons {
        width: 100%;
        height: 3;
        align: center middle;
        margin-top: 1;
    }

    Button {
        margin: 0 1;
    }
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config.copy()

    def compose(self) -> ComposeResult:
        with Container(id="repositories-container"):
            yield Label("Manage Repositories", id="title", classes="title")
            yield Label("Select repositories to process:")
            yield ListView(
                *[ListItem(Label(repo['name'])) for repo in self.config['repositories']],
                id="repo-list"
            )
            with Horizontal(id="buttons"):
                yield Button("Add", id="add", variant="primary")
                yield Button("Remove", id="remove", variant="error")
                yield Button("Save", id="save", variant="success")
                yield Button("Cancel", id="cancel")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "add":
            repo_dialog = InputDialog("Enter repository name/URL:", "")
            repo_name = await self.app.push_screen(repo_dialog)
            if repo_name:
                # Normalize repository name
                repo_name = repo_name.strip()
                if repo_name:
                    # Add repository to config and update the list view
                    self.config['repositories'].append({
                        "name": repo_name,
                        "type": "repository" if "/" in repo_name else "organization"
                    })
                    repo_list = self.query_one("#repo-list", ListView)
                    repo_list.clear()
                    # Refresh the entire list
                    for repo in self.config['repositories']:
                        repo_list.append(ListItem(Label(repo['name'])))
        elif event.button.id == "remove":
            # Get the selected item's index
            repo_list = self.query_one("#repo-list", ListView)
            selected = repo_list.index
            if selected is not None and 0 <= selected < len(self.config['repositories']):
                # Remove from config and update the list view
                del self.config['repositories'][selected]
                # Update the ListView to reflect the change
                repo_list.clear()
                for repo in self.config['repositories']:
                    repo_list.append(ListItem(Label(repo['name'])))
        elif event.button.id == "save":
            save_config(self.config)
            self.dismiss(self.config)
        elif event.button.id == "cancel":
            self.dismiss(None)


class ConfigureSettingsScreen(ModalScreen[Dict[str, Any]]):
    """Screen for configuring settings."""

    DEFAULT_CSS = """
    ConfigureSettingsScreen {
        align: center middle;
    }

    #settings-container {
        width: 60;
        height: auto;
        padding: 1 2;
        background: $surface;
        border: thick $primary;
        margin: 1;
    }

    #title {
        text-align: center;
        margin-bottom: 1;
        text-style: bold;
    }

    Label {
        margin-bottom: 1;
    }

    Input {
        margin-bottom: 1;
    }

    #buttons {
        width: 100%;
        height: 3;
        align: center middle;
        margin-top: 1;
    }

    Button {
        margin: 0 1;
    }
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config.copy()

    def compose(self) -> ComposeResult:
        with Container(id="settings-container"):
            yield Label("Configure Settings", id="title", classes="title")
            yield Label("Output Directory:")
            yield Input(value=self.config['output_directory'], placeholder="Directory to save output files", id="output-directory")
            yield Label("Test Split Ratio (0.0-1.0):")
            yield Input(value=str(self.config['test_ratio']), placeholder="Test split ratio (0.0-1.0)", id="test-ratio")
            with Horizontal(id="buttons"):
                yield Button("Save", id="save", variant="success")
                yield Button("Cancel", id="cancel", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save":
            self.config['output_directory'] = self.query_one("#output-directory", Input).value
            try:
                test_ratio = float(self.query_one("#test-ratio", Input).value)
                if 0 <= test_ratio <= 1:
                    self.config['test_ratio'] = test_ratio
                else:
                    self.app.notify("Invalid test ratio. Using default value of 0.2.", severity="warning")
                    self.config['test_ratio'] = 0.2
            except ValueError:
                self.app.notify("Invalid test ratio. Using default value of 0.2.", severity="warning")
                self.config['test_ratio'] = 0.2
            save_config(self.config)
            self.dismiss(self.config)
        else:
            self.dismiss(None)


class AutoTrainScreen(ModalScreen[Dict[str, Any]]):
    """Screen for configuring AutoTrain."""

    DEFAULT_CSS = """
    AutoTrainScreen {
        align: center middle;
    }

    #autotrain-container {
        width: 60;
        height: auto;
        padding: 1 2;
        background: $surface;
        border: thick $primary;
        margin: 1;
    }

    #title {
        text-align: center;
        margin-bottom: 1;
        text-style: bold;
    }

    Label {
        margin-bottom: 1;
    }

    Input {
        margin-bottom: 1;
    }

    #buttons {
        width: 100%;
        height: 3;
        align: center middle;
        margin-top: 1;
    }

    Button {
        margin: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Container(id="autotrain-container"):
            yield Label("AutoTrain Configuration", id="title", classes="title")
            yield Label("Model name (e.g., meta-llama/Llama-3.2-1B-Instruct):")
            yield Input(placeholder="Model name", id="model", value="meta-llama/Llama-3.2-1B-Instruct")
            yield Label("Project name:")
            yield Input(placeholder="Project name", id="project_name", value="autotrain-project")
            yield Label("Data path (e.g., HuggingFaceH4/no_robots):")
            yield Input(placeholder="Data path", id="data_path", value="HuggingFaceH4/no_robots")
            yield Label("Train split (default: train):")
            yield Input(placeholder="Train split", id="train_split", value="train")
            yield Label("Text column (default: text):")
            yield Input(placeholder="Text column", id="text_column", value="text")
            yield Label("Chat template (optional, e.g. tokenizer):")
            yield Input(placeholder="Chat template", id="chat_template", value="tokenizer")
            yield Label("Epochs (default: 3):")
            yield Input(placeholder="Epochs", id="epochs", value="3")
            yield Label("Batch size (default: 1):")
            yield Input(placeholder="Batch size", id="batch_size", value="1")
            yield Label("Learning rate (default: 1e-5):")
            yield Input(placeholder="Learning rate", id="lr", value="1e-5")
            with Horizontal(id="buttons"):
                yield Button("Train", id="train", variant="primary")
                yield Button("Cancel", id="cancel", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "train":
            # Get the values from input fields
            model = self.query_one("#model", Input).value
            project_name = self.query_one("#project_name", Input).value
            data_path = self.query_one("#data_path", Input).value
            train_split = self.query_one("#train_split", Input).value or "train"
            text_column = self.query_one("#text_column", Input).value or "text"
            chat_template = self.query_one("#chat_template", Input).value or None
            epochs = self.query_one("#epochs", Input).value or "3"
            batch_size = self.query_one("#batch_size", Input).value or "1"
            lr = self.query_one("#lr", Input).value or "1e-5"

            # Create parameters dictionary
            params = {
                "model": model,
                "project_name": project_name,
                "data_path": data_path,
                "train_split": train_split,
                "text_column": text_column,
                "chat_template": chat_template,
                "epochs": int(epochs),
                "batch_size": int(batch_size),
                "lr": float(lr),
                "peft": True,
                "quantization": "int4",
                "target_modules": "all-linear"
            }

            # Validate inputs
            if not model:
                self.app.notify("Model name is required", severity="error")
                return
            if not project_name:
                self.app.notify("Project name is required", severity="error")
                return
            if not data_path:
                self.app.notify("Data path is required", severity="error")
                return

            # Start training
            self.dismiss(params)
        else:
            self.dismiss(None)


class ConfirmationScreen(ModalScreen[bool]):
    """A modal screen that asks for confirmation."""

    DEFAULT_CSS = """
    ConfirmationScreen {
        align: center middle;
    }

    #dialog {
        width: 60;
        height: auto;
        padding: 1 2;
        background: $surface;
        border: thick $primary;
        margin: 1;
    }

    #question {
        text-align: center;
        margin-bottom: 1;
        text-style: bold;
    }

    #message {
        text-align: center;
        margin-bottom: 1;
    }

    #buttons {
        width: 100%;
        height: 3;
        align: center middle;
        margin-top: 1;
    }

    Button {
        margin: 0 1;
    }
    """

    def __init__(self, title: str) -> None:
        super().__init__()
        self.title = title

    def compose(self) -> ComposeResult:
        with Container(id="dialog"):
            yield Label(self.title, id="question")
            yield Label("Are you sure you want to exit?", id="message")
            with Horizontal(id="buttons"):
                yield Button("Yes", variant="error", id="yes")
                yield Button("No", variant="primary", id="no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "yes":
            self.dismiss(True)
        else:
            self.dismiss(False)


class MainTUIApp(App[None]):
    """Main TUI application."""

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("t", "cycle_theme", "Change theme")
    ]

    THEMES = ["monokai", "dracula", "nord", "gruvbox", "material", "default"]

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        super().__init__()
        self.title = "SDK Dataset Generator"
        # Set default theme to Monokai
        self.theme = "monokai"

    def compose(self) -> ComposeResult:
        yield Header()
        ascii_art = """
 __          ___
/  \\|_|_  _ _ | _ | _ _
\\__/|_| )(-|  |(_||(-_)

"""
        yield Label(ascii_art, id="app-title")
        with Container(id="main-container"):
            yield ListView(
                ListItem(Label("Manage Credentials"), id="credentials"),
                ListItem(Label(f"Manage Repositories ({len(self.config['repositories'])} configured)"), id="repositories"),
                ListItem(Label("Configure Settings"), id="settings"),
                ListItem(Label("Start Download and Processing"), id="start"),
                ListItem(Label("Process Existing Downloaded Data"), id="process_existing"),
                ListItem(Label("Train with AutoTrain"), id="autotrain"),
                ListItem(Label("Upload Existing Dataset to Hugging Face Hub"), id="upload"),
                ListItem(Label("Exit"), id="exit"),
                id="menu-list"
            )
        yield Footer()

    def action_cycle_theme(self) -> None:
        """Cycle through available themes."""
        current_index = self.THEMES.index(self.theme) if self.theme in self.THEMES else 0
        next_index = (current_index + 1) % len(self.THEMES)
        self.theme = self.THEMES[next_index]
        self.notify(f"Theme changed to: {self.theme}")

    @work
    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        choice = event.item.id

        if choice == "credentials":
            screen = ManageCredentialsScreen(self.config)
            updated_config = await self.push_screen(screen)
            if updated_config:
                self.config = updated_config

        elif choice == "repositories":
            screen = ManageRepositoriesScreen(self.config)
            updated_config = await self.push_screen(screen)
            if updated_config:
                self.config = updated_config
                # Update the repositories count in the menu
                repo_count = len(self.config['repositories'])
                self.query_one("#menu-list").clear()
                self.query_one("#menu-list").append(ListItem(Label("Manage Credentials"), id="credentials"))
                self.query_one("#menu-list").append(ListItem(Label(f"Manage Repositories ({repo_count} configured)"), id="repositories"))
                self.query_one("#menu-list").append(ListItem(Label("Configure Settings"), id="settings"))
                self.query_one("#menu-list").append(ListItem(Label("Start Download and Processing"), id="start"))
                self.query_one("#menu-list").append(ListItem(Label("Process Existing Downloaded Data"), id="process_existing"))
                self.query_one("#menu-list").append(ListItem(Label("Train with AutoTrain"), id="autotrain"))
                self.query_one("#menu-list").append(ListItem(Label("Upload Existing Dataset to Hugging Face Hub"), id="upload"))
                self.query_one("#menu-list").append(ListItem(Label("Exit"), id="exit"))

        elif choice == "settings":
            screen = ConfigureSettingsScreen(self.config)
            updated_config = await self.push_screen(screen)
            if updated_config:
                self.config = updated_config

        elif choice == "start":
            if not self.config['repositories']:
                await self.push_screen(MessageDialog("Error", "No repositories configured. Please add repositories first."))
                return

            # Start processing
            self.notify("Starting download and processing...", severity="information")

            # Run the processing in a worker to avoid freezing
            result = await self.run_worker(
                process_all_repositories,
                self.config
            )

            if result:
                dataset, sft_dataset, code_gen_dataset = result
                self.notify("Processing completed successfully.", severity="success")

                # Ask if user wants to upload to Hugging Face Hub
                upload_dialog = SelectDialog(
                    "Upload Dataset?",
                    "Do you want to upload this dataset to Hugging Face Hub?",
                    [
                        ("standard", "Yes, upload standard dataset"),
                        ("sft", "Yes, upload SFT-ready dataset"),
                        ("code_gen", "Yes, upload code generation dataset"),
                        ("all", "Yes, upload all datasets"),
                        ("no", "No, skip upload")
                    ]
                )
                upload_choice = await self.push_screen(upload_dialog)

                if upload_choice and upload_choice != "no":
                    # Handle uploading based on choice
                    if upload_choice == "standard" or upload_choice == "all":
                        name_dialog = InputDialog("Enter dataset name for Standard dataset:")
                        name = await self.push_screen(name_dialog)
                        await self.run_worker(
                            lambda: upload_to_huggingface(dataset, self.config, "standard", name)
                        )
                    if upload_choice == "sft" or upload_choice == "all":
                        name_dialog = InputDialog("Enter dataset name for SFT dataset:")
                        name = await self.push_screen(name_dialog)
                        await self.run_worker(
                            lambda: upload_to_huggingface(sft_dataset, self.config, "sft", name)
                        )
                    if upload_choice == "code_gen" or upload_choice == "all":
                        name_dialog = InputDialog("Enter dataset name for Code Generation dataset:")
                        name = await self.push_screen(name_dialog)
                        await self.run_worker(
                            lambda: upload_to_huggingface(code_gen_dataset, self.config, "code_gen", name)
                        )
            else:
                await self.push_screen(MessageDialog("Error", "Failed to create dataset. Check logs for details."))

        elif choice == "process_existing":
            # Ask for the directory with existing data
            dir_dialog = InputDialog(
                "Enter the directory path containing the downloaded data:",
                default_value=self.config['output_directory']
            )
            dir_path = await self.push_screen(dir_dialog)

            if dir_path:
                # Ask for test ratio
                ratio_dialog = InputDialog(
                    "Enter the test split ratio (between 0 and 1):",
                    default_value=str(self.config['test_ratio'])
                )
                test_ratio_str = await self.push_screen(ratio_dialog)

                test_ratio = self.config['test_ratio']
                if test_ratio_str:
                    try:
                        test_ratio = float(test_ratio_str)
                        if not 0 <= test_ratio <= 1:
                            test_ratio = self.config['test_ratio']
                            self.notify(f"Invalid test ratio. Using default: {test_ratio}", severity="warning")
                    except ValueError:
                        self.notify(f"Invalid test ratio. Using default: {test_ratio}", severity="warning")

                # Process the existing data
                self.notify(f"Processing existing data in {dir_path} with test ratio {test_ratio}...", severity="information")

                # Run in worker to avoid freezing
                result = await self.run_worker(
                    process_existing_data,
                    dir_path, test_ratio
                )

                if result:
                    dataset, sft_dataset, code_gen_dataset = result
                    self.notify("Processing of existing data completed successfully.", severity="success")

                    # Ask if user wants to upload to Hugging Face Hub
                    upload_dialog = SelectDialog(
                        "Upload Dataset?",
                        "Do you want to upload this dataset to Hugging Face Hub?",
                        [
                            ("standard", "Yes, upload standard dataset"),
                            ("sft", "Yes, upload SFT-ready dataset"),
                            ("code_gen", "Yes, upload code generation dataset"),
                            ("all", "Yes, upload all datasets"),
                            ("no", "No, skip upload")
                        ]
                    )
                    upload_choice = await self.push_screen(upload_dialog)

                    if upload_choice and upload_choice != "no":
                        if upload_choice == "standard" or upload_choice == "all":
                            name_dialog = InputDialog("Enter dataset name for Standard dataset:")
                            name = await self.push_screen(name_dialog)
                            await self.run_worker(
                                lambda: upload_to_huggingface(dataset, self.config, "standard", name)
                            )
                        if upload_choice == "sft" or upload_choice == "all":
                            name_dialog = InputDialog("Enter dataset name for SFT dataset:")
                            name = await self.push_screen(name_dialog)
                            await self.run_worker(
                                lambda: upload_to_huggingface(sft_dataset, self.config, "sft", name)
                            )
                        if upload_choice == "code_gen" or upload_choice == "all":
                            name_dialog = InputDialog("Enter dataset name for Code Generation dataset:")
                            name = await self.push_screen(name_dialog)
                            await self.run_worker(
                                lambda: upload_to_huggingface(code_gen_dataset, self.config, "code_gen", name)
                            )
                else:
                    await self.push_screen(MessageDialog("Error", "Failed to process existing data. Check logs for details."))

        elif choice == "autotrain":
            # Configure AutoTrain
            autotrain_params = await self.push_screen(AutoTrainScreen())
            if autotrain_params:
                # Start training with AutoTrain
                result = await self.run_worker(self.run_autotrain, autotrain_params)
                if result:
                    self.notify("Training completed successfully.", severity="success")
                else:
                    await self.push_screen(MessageDialog("Error", "Failed to start training. Check logs for details."))

        elif choice == "upload":
            # Ask which type of dataset to upload
            type_dialog = SelectDialog(
                "Dataset Type",
                "Which type of dataset do you want to upload?",
                [
                    ("standard", "Standard Dataset"),
                    ("sft", "SFT-Ready Dataset"),
                    ("code_gen", "Code Generation Dataset")
                ]
            )
            dataset_type = await self.push_screen(type_dialog)

            if dataset_type:
                # Determine path based on dataset type
                path_suffix = {
                    "standard": "dataset",
                    "sft": "sft_dataset",
                    "code_gen": "code_generation_dataset"
                }.get(dataset_type, "dataset")

                default_path = os.path.join(self.config['output_directory'], path_suffix)

                path_dialog = InputDialog(
                    f"Enter path to the {dataset_type.replace('_', ' ').title()} dataset:",
                    default_value=default_path
                )
                dataset_path = await self.push_screen(path_dialog)

                if dataset_path:
                    try:
                        # Load the dataset
                        self.notify(f"Loading dataset from {dataset_path}...", severity="information")

                        # Run loading in worker to avoid freezing
                        dataset = await self.run_worker(
                            lambda: DatasetDict.load_from_disk(dataset_path),
                            thread=True
                        )

                        self.notify(f"Dataset loaded with {len(dataset['train'])} training samples", severity="success")
                        if 'test' in dataset:
                            self.notify(f"and {len(dataset['test'])} testing samples", severity="success")

                        # Upload the dataset
                        name_dialog = InputDialog("Enter a name for the dataset on Hugging Face Hub:")
                        dataset_name = await self.push_screen(name_dialog)

                        # Run upload in worker to avoid freezing
                        await self.run_worker(
                            lambda: upload_to_huggingface(dataset, self.config, dataset_type, dataset_name)
                        )
                    except Exception as e:
                        await self.push_screen(MessageDialog("Error", f"Error loading dataset: {str(e)}"))

        elif choice == "exit":
            # Create a proper confirmation dialog
            confirm_screen = ConfirmationScreen("Exit Application")
            confirmed = await self.push_screen(confirm_screen)
            if confirmed:
                await self.action_quit()

    @work
    async def run_autotrain(self, params: Dict[str, Any]) -> bool:
        """Run AutoTrain with the provided parameters."""
        try:
            self.notify(f"Starting AutoTrain with model {params['model']}...", severity="information")

            # Import autotrain
            try:
                from autotrain.params import LLMTrainingParams
                from autotrain.project import AutoTrainProject
            except ImportError:
                self.notify("AutoTrain not installed. Installing...", severity="information")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "autotrain-advanced"])
                from autotrain.params import LLMTrainingParams
                from autotrain.project import AutoTrainProject

            # Create parameters
            train_params = LLMTrainingParams(
                model=params["model"],
                project_name=params["project_name"],
                data_path=params["data_path"],
                train_split=params["train_split"],
                text_column=params["text_column"],
                chat_template=params.get("chat_template"),
                epochs=params["epochs"],
                batch_size=params["batch_size"],
                lr=params["lr"],
                peft=params.get("peft", True),
                quantization=params.get("quantization", "int4"),
                target_modules=params.get("target_modules", "all-linear"),
                username=os.environ.get("HF_USERNAME"),
                token=os.environ.get("HF_TOKEN", self.config.get('huggingface_token', '')),
            )

            # Create and run project
            project = AutoTrainProject(params=train_params, backend="local", process=True)
            project.create()

            return True
        except Exception as e:
            self.notify(f"Error running AutoTrain: {str(e)}", severity="error")
            return False

    def action_quit(self) -> None:
        self.push_screen(
            ConfirmationScreen("Are you sure you want to quit?"),
            lambda result: self.exit() if result else None
        )

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

def get_organization_repos_github(organization: str, token: Optional[str] = None, 
                                retry_limit: int = 3, retry_delay: int = 60) -> List[Dict[str, Any]]:
    """Enhanced version with better rate limit handling"""
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
    """
    repo_type = repo_info.get('type', 'repository')
    # Always use GitHub as source
    source = 'github'
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
                repo_name = repo_info['name'].split('/')[-1]
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

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='SDK Documentation Dataset Generator',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--no-tui', action='store_true', 
                       help='Run in command line mode without TUI')
    parser.add_argument('--org', type=str, 
                       help='Organization to process')
    parser.add_argument('--output', type=str, 
                       default='./downloaded_docs',
                       help='Output directory')
    parser.add_argument('--test-ratio', type=float, 
                       default=0.2,
                       help='Test split ratio')
    parser.add_argument('--github-token', type=str,
                       default=os.getenv('GITHUB_TOKEN'),
                       help='GitHub API token (or set GITHUB_TOKEN env var)')
    parser.add_argument('--hf-token', type=str,
                       default=os.getenv('HF_TOKEN'),
                       help='Hugging Face API token (or set HF_TOKEN env var)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--retry-limit', type=int,
                       default=3,
                       help='Number of retries for API calls')
    parser.add_argument('--retry-delay', type=int,
                       default=60,
                       help='Delay in seconds between retries')
    return parser.parse_args()

def setup_logging(debug: bool = False) -> None:
    """
    Set up logging configuration.

    Args:
        debug (bool): Enable debug logging if True
    """
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("doc_downloader.log"),
            logging.StreamHandler()
        ]
    )

def main_tui() -> int:
    """
    Main function with TUI interface.

    Returns:
        int: Exit code (0 for success, non-zero for errors)
    """
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Set up logging
        setup_logging(args.debug)
        
        # Load or create configuration
        config = load_or_create_config()
        
        # Update config with command line arguments if provided
        if args.github_token:
            config['github_token'] = args.github_token
        if args.hf_token:
            config['huggingface_token'] = args.hf_token
        if args.output:
            config['output_directory'] = args.output
        if args.test_ratio:
            config['test_ratio'] = args.test_ratio
            
        # If organization is provided via command line, add it to repositories
        if args.org:
            config['repositories'].append({
                'type': 'organization',
                'name': args.org,
                'source': 'github'
            })
            
        # Save updated config
        save_config(config)
        
        # If no-tui flag is set and org is provided, run in command line mode
        if args.no_tui and args.org:
            logger.info(f"Running in command line mode for organization: {args.org}")
            result = process_all_repositories(config)
            return 0 if result else 1
            
        # Otherwise, start TUI
        app = MainTUIApp(config)
        app.run()
        return 0
        
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.debug(traceback.format_exc())
        return 1

def main() -> int:
    """
    Legacy main function (kept for backward compatibility).
    
    Returns:
        int: Exit code (0 for success, non-zero for errors)
    """
    try:
        print("Information: This script now uses a text-based user interface (TUI). Switching to TUI mode...")
        return main_tui()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
