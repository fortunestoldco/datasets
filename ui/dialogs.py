#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import List, Tuple
from textual import on
from textual.app import ComposeResult
from textual.widgets import Button, Input, Label, ListView, ListItem
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen

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
