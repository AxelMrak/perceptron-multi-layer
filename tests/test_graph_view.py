"""Tests for GraphView — the passive matplotlib decision-boundary viewer."""

import matplotlib

matplotlib.use("Agg")

import tkinter as tk

import pytest
from view.graph_view import GraphView


@pytest.fixture
def gv():
    """Create a GraphView with a hidden tkinter root window."""
    root = tk.Tk()
    root.withdraw()
    view = GraphView(root)
    yield view
    root.destroy()


class TestInit:
    def test_figure_and_axes_created(self, gv):
        assert gv._figure is not None
        assert gv._axes is not None

    def test_axes_limits(self, gv):
        assert gv._axes.get_xlim() == (-2, 2)
        assert gv._axes.get_ylim() == (-2, 2)

    def test_four_points_plotted_at_start(self, gv):
        assert len(gv._axes.lines) == 4


class TestPlotDecisionBoundary:
    def test_adds_one_line_to_axes(self, gv):
        n_before = len(gv._axes.lines)
        gv.plot_decision_boundary(0.0, 1.0, 1.0, epoch=1)
        assert len(gv._axes.lines) == n_before + 1


class TestPlotDecisionBoundaryLabel:
    def test_line_has_correct_epoch_label(self, gv):
        gv.plot_decision_boundary(0.0, 1.0, 1.0, epoch=5)
        new_line = gv._axes.lines[-1]
        assert new_line.get_label() == "Epoch 5"


class TestClear:
    def test_removes_decision_line_leaves_points(self, gv):
        gv.plot_decision_boundary(0.0, 1.0, 1.0, epoch=1)
        gv.clear()
        assert len(gv._axes.lines) == 4

    def test_redraws_four_points_after_clear(self, gv):
        gv.plot_decision_boundary(0.0, 1.0, 1.0, epoch=1)
        gv.clear()
        colors = {line.get_color() for line in gv._axes.lines}
        assert colors == {"red", "green", "yellow", "blue"}

class TestClassConstants:
    def test_figure_size(self):
        assert GraphView.FIGURE_SIZE == (5, 4)

    def test_dpi(self):
        assert GraphView.DPI == 100

    def test_axis_bounds(self):
        assert GraphView.AXIS_MIN == -2
        assert GraphView.AXIS_MAX == 2

    def test_line_style(self):
        assert GraphView.LINE_COLOR == "purple"
        assert GraphView.LINE_STYLE == "--"
