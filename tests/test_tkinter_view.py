import tkinter as tk
from tkinter import ttk
from unittest.mock import patch

import pytest

from view.tkinter_view import TkinterView


@pytest.fixture(scope="module")
def tk_window():
    window = tk.Tk()
    yield window
    window.destroy()


class TestInit:
    def test_label_attributes(self, tk_window):
        view = TkinterView(tk_window)
        for name in [
            "title_label",
            "input1_label",
            "input2_label",
            "weight1_label",
            "weight2_label",
            "bias_label",
            "desired_label",
            "obtained_label",
            "test_result_label",
        ]:
            assert isinstance(getattr(view, name), tk.Label), f"{name} is not a Label"

    def test_entry_attributes(self, tk_window):
        view = TkinterView(tk_window)
        assert isinstance(view.test_input1, tk.Entry)
        assert isinstance(view.test_input2, tk.Entry)

    def test_button_attributes(self, tk_window):
        view = TkinterView(tk_window)
        assert isinstance(view.train_button, ttk.Button)
        assert isinstance(view.learn_button, ttk.Button)
        assert isinstance(view.test_button, ttk.Button)


class TestUpdateTrainingDisplay:
    def test_shows_epoch_when_not_converged(self, tk_window):
        view = TkinterView(tk_window)
        view.update_training_display(
            5, False, "1", "-1", "0.5", "-0.3", "0.8", "1", "-1"
        )
        assert view.title_label.cget("text") == "EPOCH 5"
        assert view.input1_label.cget("text") == "Input 1: 1"
        assert view.input2_label.cget("text") == "Input 2: -1"
        assert view.weight1_label.cget("text") == "Weight 1: 0.5"
        assert view.weight2_label.cget("text") == "Weight 2: -0.3"
        assert view.bias_label.cget("text") == "Bias weight: 0.8"
        assert view.desired_label.cget("text") == "Desired output: 1"
        assert view.obtained_label.cget("text") == "Obtained output: -1"

    def test_shows_complete_when_converged(self, tk_window):
        view = TkinterView(tk_window)
        view.update_training_display(
            10, True, "1", "1", "1.0", "1.0", "0.0", "1", "1"
        )
        assert view.title_label.cget("text") == "TRAINING COMPLETE — FINAL WEIGHTS:"
        assert view.input1_label.cget("text") == "Input 1: 1"

    def test_subsequent_calls_override_values(self, tk_window):
        view = TkinterView(tk_window)
        view.update_training_display(1, False, "1", "-1", "0", "0", "0", "0", "0")
        view.update_training_display(2, False, "-1", "1", "1", "1", "1", "1", "1")
        assert view.title_label.cget("text") == "EPOCH 2"
        assert view.input1_label.cget("text") == "Input 1: -1"
        assert view.input2_label.cget("text") == "Input 2: 1"


class TestUpdateTestResult:
    def test_updates_result_label(self, tk_window):
        view = TkinterView(tk_window)
        view.update_test_result("Result: 1 — True")
        assert view.test_result_label.cget("text") == "Result: 1 — True"

    def test_overwrites_previous_result(self, tk_window):
        view = TkinterView(tk_window)
        view.update_test_result("first")
        view.update_test_result("second")
        assert view.test_result_label.cget("text") == "second"


class TestShowError:
    def test_calls_showerror_with_correct_args(self, tk_window):
        view = TkinterView(tk_window)
        with patch("tkinter.messagebox.showerror") as mock:
            view.show_error("Something went wrong")
        mock.assert_called_once_with("Error", "Something went wrong")


class TestShowInfo:
    def test_calls_showinfo_with_correct_args(self, tk_window):
        view = TkinterView(tk_window)
        with patch("tkinter.messagebox.showinfo") as mock:
            view.show_info("Training complete")
        mock.assert_called_once_with("Info", "Training complete")


class TestEntryWidgets:
    def test_insert_and_get(self, tk_window):
        view = TkinterView(tk_window)
        view.test_input1.insert(0, "1")
        view.test_input2.insert(0, "-1")
        assert view.test_input1.get() == "1"
        assert view.test_input2.get() == "-1"
