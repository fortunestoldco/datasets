#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Dict, Any
from textual import on
from textual.app import ComposeResult
from textual.widgets import Button, Input, Label, ListView, ListItem
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen

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
            from config import save_config
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
        # Store repositories as regular list instead of reactive
        self._repositories = list(self.config['repositories'])
        self._filtered_repositories = list(self._repositories)

    def compose(self) -> ComposeResult:
        with Container(id="repositories-container"):
            yield Label("Manage Repositories", id="title", classes="title")
            yield Input(placeholder="Search repositories...", id="search-bar")
            yield Label("Select repositories to process:")
            # Initialize ListView with current repositories
            yield ListView(
                *[ListItem(Label(f"{repo['name']} ({repo['type']}, {repo['source']}, {repo['num_files']} files)")) for repo in self._filtered_repositories],
                id="repo-list"
            )
            with Horizontal(id="buttons"):
                yield Button("Add", id="add", variant="primary")
                yield Button("Remove", id="remove", variant="error")
                yield Button("Save", id="save", variant="success")
                yield Button("Cancel", id="cancel")

    def update_repo_list(self):
        """Helper method to update the repository list display"""
        repo_list = self.query_one("#repo-list", ListView)
        repo_list.clear()
        # Add all current repositories to ListView
        for repo in self._filtered_repositories:
            repo_list.append(ListItem(Label(f"{repo['name']} ({repo['type']}, {repo['source']}, {repo['num_files']} files)")))
        self.refresh(repaint=True)

    def filter_repositories(self, query: str) -> None:
        """Filter repositories based on the search query"""
        self._filtered_repositories = [repo for repo in self._repositories if query.lower() in repo['name'].lower() or query.lower() in repo['type'].lower() or query.lower() in repo['source'].lower()]
        self.update_repo_list()

    @on(Input.Changed, "#search-bar")
    def on_search_bar_changed(self, event: Input.Changed) -> None:
        self.filter_repositories(event.value)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "add":
            from ui.dialogs import InputDialog
            repo_dialog = InputDialog("Enter repository name/URL:", "")
            repo_name = await self.app.push_screen(repo_dialog)
            if repo_name:
                # Normalize repository name
                repo_name = repo_name.strip()
                if repo_name:
                    # Add to internal list
                    self._repositories.append({
                        "name": repo_name,
                        "type": "repository" if "/" in repo_name else "organization",
                        "source": "github" if "github.com" in repo_name else "huggingface",
                        "num_files": 0  # Placeholder for number of files
                    })
                    # Update internal lists
                    self._filtered_repositories = list(self._repositories)
                    # Update ListView
                    self.update_repo_list()
                    from config import save_config
                    save_config(self.config)

        elif event.button.id == "remove":
            repo_list = self.query_one("#repo-list", ListView)
            selected = repo_list.index
            if selected is not None and 0 <= selected < len(self._filtered_repositories):
                # Remove from internal list
                del self._repositories[selected]
                # Update internal lists
                self._filtered_repositories = list(self._repositories)
                # Update ListView
                self.update_repo_list()
                from config import save_config
                save_config(self.config)

        elif event.button.id == "save":
            # Update config before saving
            self.config['repositories'] = list(self._repositories)
            from config import save_config
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
            from config import save_config
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
        width: 120;  /* Increased width to accommodate two columns */
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
        width: 100%;
    }

    .column {
        width: 50%;
        height: auto;
        padding: 0 1;
    }

    Label {
        margin-bottom: 1;
        text-align: left;
    }

    Input {
        margin-bottom: 1;
        width: 100%;
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
            
            with Horizontal():
                # Left column
                with Vertical(classes="column"):
                    yield Label("Model name:")
                    yield Input(placeholder="Model name", id="model", value="meta-llama/Llama-3.2-1B-Instruct")
                    yield Label("Project name:")
                    yield Input(placeholder="Project name", id="project_name", value="autotrain-project")
                    yield Label("Data path:")
                    yield Input(placeholder="Data path", id="data_path", value="HuggingFaceH4/no_robots")
                    yield Label("Train split:")
                    yield Input(placeholder="Train split", id="train_split", value="train")
                    yield Label("Text column:")
                    yield Input(placeholder="Text column", id="text_column", value="text")
                
                # Right column
                with Vertical(classes="column"):
                    yield Label("Chat template:")
                    yield Input(placeholder="Chat template", id="chat_template", value="tokenizer")
                    yield Label("Epochs:")
                    yield Input(placeholder="Epochs", id="epochs", value="3")
                    yield Label("Batch size:")
                    yield Input(placeholder="Batch size", id="batch_size", value="1")
                    yield Label("Learning rate:")
                    yield Input(placeholder="Learning rate", id="lr", value="1e-5")
            
            with Horizontal(id="buttons"):
                yield Button("Train", id="train", variant="primary")
                yield Button("Cancel", id="cancel", variant="error")
