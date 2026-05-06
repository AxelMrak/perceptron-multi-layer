import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class GraphView:
    """
    Passive View for the decision boundary graph.

    Draws the 4 AND gate points and the decision boundary line.
    Updated when weights change.
    """

    # Display constants
    FIGURE_SIZE = (5, 4)
    DPI = 100
    AXIS_MIN = -2
    AXIS_MAX = 2
    LINE_COLOR = "purple"
    LINE_STYLE = "--"

    def __init__(self, window) -> None:
        """
        Initialize the matplotlib figure embedded in tkinter.

        Args:
            window: The root tkinter window.
        """
        self._figure = plt.figure(figsize=self.FIGURE_SIZE, dpi=self.DPI)
        self._axes = self._figure.add_subplot(111)

        self._setup_axes()
        self._plot_points()

        self._canvas = FigureCanvasTkAgg(self._figure, master=window)
        self._canvas.draw()
        self._canvas.get_tk_widget().grid(row=7, column=2, columnspan=5)

    def _setup_axes(self) -> None:
        """Configure axis: grid, limits, spine positions at origin."""
        self._axes.grid(True)
        self._axes.set_xlim(self.AXIS_MIN, self.AXIS_MAX)
        self._axes.set_ylim(self.AXIS_MIN, self.AXIS_MAX)

        self._axes.spines["left"].set_position("zero")
        self._axes.spines["bottom"].set_position("zero")
        self._axes.spines["right"].set_color("none")
        self._axes.spines["top"].set_color("none")

    def _plot_points(self) -> None:
        """
        Plot the 4 AND gate input combinations.

        (1, 1)   → red    (should output 1)
        (1, -1)  → green  (should output -1)
        (-1, 1)  → yellow (should output -1)
        (-1, -1) → blue   (should output -1)
        """
        self._axes.plot(1, 1, "ro")
        self._axes.plot(1, -1, "go")
        self._axes.plot(-1, 1, "yo")
        self._axes.plot(-1, -1, "bo")

    def plot_decision_boundary(
        self,
        bias_weight: float,
        weight1: float,
        weight2: float,
        epoch: int,
    ) -> None:
        """
        Draw the decision boundary line for current weights.

        Line equation: bias + w1*x1 + w2*x2 = 0
        Solved for x2: x2 = (-bias - w1*x1) / w2
        """
        y_at_xmin = (-bias_weight - weight1 * self.AXIS_MIN) / weight2
        y_at_xmax = (-bias_weight - weight1 * self.AXIS_MAX) / weight2

        self._axes.plot(
            [self.AXIS_MIN, self.AXIS_MAX],
            [y_at_xmin, y_at_xmax],
            color=self.LINE_COLOR,
            linestyle=self.LINE_STYLE,
            label=f"Epoch {epoch}",
        )
        self._axes.legend(loc="upper right")
        self._canvas.draw()

    def clear(self) -> None:
        """Clear the graph and redraw base points."""
        self._axes.clear()
        self._setup_axes()
        self._plot_points()
        self._canvas.draw()
