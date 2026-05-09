import tkinter as tk
from tkinter import ttk


class TkinterView:
    """
    Passive View for the perceptron trainer.

    Builds all widgets and layout. Exposes public attributes for the controller.
    """

    def __init__(self, window: tk.Tk) -> None:
        """
        Build the complete UI.

        Args:
            window: The root tkinter window (created in main.py).
        """
        window.geometry("900x700")
        window.title("PERCEPTRON SIMPLE — AND")

        #  Style
        title_font = ("Helvetica", 10, "bold")
        button_style = ttk.Style()
        button_style.configure(
            "Action.TButton",
            borderwidth=0,
            relief="flat",
            font=("Helvetica", 10, "bold"),
        )

        #  Info labels
        self.title_label = tk.Label(
            window,
            text="Click 'Train' to begin",
            font=title_font,
            pady=10,
        )
        self.input1_label = tk.Label(window, text="Input 1:", pady=10, padx=90)
        self.input2_label = tk.Label(window, text="Input 2:")
        bias_label = tk.Label(window, text="Bias (input 0): 1")

        self.weight1_label = tk.Label(window, text="Weight 1:", pady=10)
        self.weight2_label = tk.Label(window, text="Weight 2:")
        self.bias_label = tk.Label(window, text="Bias weight:")

        self.desired_label = tk.Label(window, text="Desired output:", pady=10)
        self.obtained_label = tk.Label(window, text="Obtained output:")

        learning_rate_label = tk.Label(window, text="Learning rate: 0.6")

        #  Test section
        test_title = tk.Label(
            window,
            text="TEST THE PERCEPTRON:",
            font=title_font,
            pady=10,
        )
        test_input1_label = tk.Label(window, text="Input 1:", pady=10)
        test_input2_label = tk.Label(window, text="Input 2:")

        self.test_input1 = tk.Entry(window)
        self.test_input2 = tk.Entry(window)

        self.test_result_label = tk.Label(window, text="Result:", pady=10)

        #  Buttons (wired by main.py)
        self.train_button = ttk.Button(
            window,
            text="Train",
            style="Action.TButton",
            width=15,
        )
        self.learn_button = ttk.Button(
            window,
            text="Learn Step",
            style="Action.TButton",
            width=15,
        )
        self.test_button = ttk.Button(
            window,
            text="Test",
            style="Action.TButton",
            width=15,
        )

        #  Spacers
        spacer1 = tk.Label(window, text="", padx=15)
        spacer2 = tk.Label(window, text="", padx=15)
        spacer3 = tk.Label(window, text="", padx=15)

        #  Layout (grid)
        # Row 0: Title
        self.title_label.grid(row=0, column=3)

        # Row 1: Training info
        spacer1.grid(row=1, column=0)
        self.train_button.grid(row=1, column=1)
        self.input1_label.grid(row=1, column=2)
        self.input2_label.grid(row=1, column=3)
        bias_label.grid(row=1, column=4)

        # Row 2: Weights
        self.weight1_label.grid(row=2, column=2)
        self.weight2_label.grid(row=2, column=3)
        self.bias_label.grid(row=2, column=4)

        # Row 3: Output info
        spacer2.grid(row=3, column=0)
        self.learn_button.grid(row=3, column=1)
        self.desired_label.grid(row=3, column=2)
        self.obtained_label.grid(row=3, column=3)
        learning_rate_label.grid(row=3, column=4)

        # Row 4: Test section title
        test_title.grid(row=4, column=3)

        # Row 5: Test inputs
        spacer3.grid(row=5, column=0)
        self.test_button.grid(row=5, column=1)
        test_input1_label.grid(row=5, column=2)
        test_input2_label.grid(row=5, column=3)

        # Row 6: Test entry fields + result
        self.test_input1.grid(row=6, column=2)
        self.test_input2.grid(row=6, column=3)
        self.test_result_label.grid(row=6, column=4)

    #  Public update methods (called by controller)
    def update_training_display(
        self,
        epoch: int,
        converged: bool,
        sample_input1: str,
        sample_input2: str,
        weight1: str,
        weight2: str,
        bias_weight: str,
        desired: str,
        obtained: str,
    ) -> None:
        """
        Refresh training info labels with current state.
        """
        self.title_label.config(
            text="TRAINING COMPLETE — FINAL WEIGHTS:"
            if converged
            else f"EPOCH {epoch}"
        )
        self.input1_label.config(text=f"Input 1: {sample_input1}")
        self.input2_label.config(text=f"Input 2: {sample_input2}")
        self.weight1_label.config(text=f"Weight 1: {weight1}")
        self.weight2_label.config(text=f"Weight 2: {weight2}")
        self.bias_label.config(text=f"Bias weight: {bias_weight}")
        self.desired_label.config(text=f"Desired output: {desired}")
        self.obtained_label.config(text=f"Obtained output: {obtained}")

    def update_test_result(self, result_text: str) -> None:
        """Show prediction result in the test section."""
        self.test_result_label.config(text=result_text)

    def show_error(self, message: str) -> None:
        """Show an error popup (e.g., invalid input)."""
        from tkinter import messagebox
        messagebox.showerror("Error", message)

    def show_info(self, message: str) -> None:
        """Show an info popup (e.g., weights after learning step)."""
        from tkinter import messagebox
        messagebox.showinfo("Info", message)
