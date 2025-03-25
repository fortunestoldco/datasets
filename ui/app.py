#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import subprocess
from typing import Dict, Any, Optional, Tuple
from textual import on, work
from textual.app import App, ComposeResult
from textual.widgets import Button, Label, ListView, ListItem, Header, Footer
from textual.containers import Container

from ui.dialogs import MessageDialog, InputDialog, SelectDialog, ConfirmationScreen
from ui.screens import ManageCredentialsScreen, ManageRepositoriesScreen, ConfigureSettingsScreen, AutoTrainScreen
from processors import process_all_repositories, process_existing_data
from dataset_utils import upload_to_huggingface
from config import save_config

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
        repo_count = len(self.config.get('repositories', []))
        with Container(id="main-container"):
            yield ListView(
                ListItem(Label("Manage Credentials"), id="credentials"),
                ListItem(Label(f"Manage Repositories ({repo_count} configured)"), id="repositories"),
                ListItem(Label("Configure Settings"), id="settings"),
                ListItem(Label("Generate new Dataset"), id="start"),
                ListItem(Label("Process Existing Downloaded Data"), id="process_existing"),
                ListItem(Label("Train with AutoTrain"), id="autotrain"),
                ListItem(Label("Upload Existing Dataset to Hugging Face Hub"), id="upload"),
                ListItem(Label("Exit"), id="exit"),
                id="menu-list"
            )
        yield Footer()

    def _update_main_menu(self):
        """Helper method to update the main menu"""
        repo_count = len(self.config['repositories'])
        menu_list = self.query_one("#menu-list")
        menu_list.clear()
        menu_list.append(ListItem(Label("Manage Credentials"), id="credentials"))
        menu_list.append(ListItem(Label(f"Manage Repositories ({repo_count} configured)"), id="repositories"))
        menu_list.append(ListItem(Label("Configure Settings"), id="settings"))
        menu_list.append(ListItem(Label("Generate new Dataset"), id="start"))
        menu_list.append(ListItem(Label("Process Existing Downloaded Data"), id="process_existing"))
        menu_list.append(ListItem(Label("Train with AutoTrain"), id="autotrain"))
        menu_list.append(ListItem(Label("Upload Existing Dataset to Hugging Face Hub"), id="upload"))
        menu_list.append(ListItem(Label("Exit"), id="exit"))

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
                self._update_main_menu()

        elif choice == "repositories":
            screen = ManageRepositoriesScreen(self.config)
            updated_config = await self.push_screen(screen)
            if updated_config:
                self.config = updated_config
                self._update_main_menu()

        elif choice == "settings":
            screen = ConfigureSettingsScreen(self.config)
            updated_config = await self.push_screen(screen)
            if updated_config:
                self.config = updated_config
                self._update_main_menu()

        elif choice == "start":
            # Show input dialog for repository URL
            url_dialog = InputDialog("Enter repository URL or organization/name:", "")
            repo_url = await self.push_screen(url_dialog)
            
            if repo_url:
                # Clean up the URL
                repo_url = repo_url.strip()
                if not repo_url:
                    await self.push_screen(MessageDialog("Error", "Please enter a valid repository URL or name."))
                    return

                # Format repo name if it's a shorthand (e.g. "org/repo")
                if '/' in repo_url and not repo_url.startswith(('http://', 'https://')):
                    if not repo_url.startswith('github.com/'):
                        repo_url = f"https://github.com/{repo_url}"
                    else:
                        repo_url = f"https://{repo_url}"

                # Show confirmation dialog
                confirm = await self.push_screen(ConfirmationScreen(
                    "Generate Dataset", 
                    f"Generate dataset from {repo_url}?"
                ))
                
                if confirm:
                    # Configure single repository
                    self.config['repositories'] = [{
                        "name": repo_url,
                        "type": "repository",
                        "source": "github",
                        "url": repo_url,  # Add the URL explicitly
                        "num_files": 0
                    }]

                    # Save the config to ensure it persists
                    save_config(self.config)
                    self._update_main_menu()
                    
                    # Start processing
                    self.notify("Starting dataset generation...", severity="information")

                    # Run the processing in a worker
                    result = await self.run_worker(
                        process_all_repositories,
                        self.config
                    )

                    if result:
                        dataset, sft_dataset, code_gen_dataset = result
                        self.notify("Dataset generation completed successfully.", severity="success")

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
            autotrain_screen = AutoTrainScreen()
            autotrain_params = await self.push_screen(autotrain_screen)
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
                        from datasets import DatasetDict
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
            confirm_screen = ConfirmationScreen("Exit Application", "Are you sure you want to quit?")
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
            ConfirmationScreen("Exit Application", "Are you sure you want to quit?"),
            lambda result: self.exit() if result else None
        )