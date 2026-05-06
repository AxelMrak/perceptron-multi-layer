# Perceptron
LEARNING_RATE = 0.6

# AND gate truth table (bias term is first element)
AND_INPUTS = [
    [1.0, 1.0, 1.0],
    [1.0, 1.0, -1.0],
    [1.0, -1.0, 1.0],
    [1.0, -1.0, -1.0],
]
AND_OUTPUTS = [1.0, -1.0, -1.0, -1.0]

# Graph
GRAPH_X_MIN = -2
GRAPH_X_MAX = 2
GRAPH_Y_MIN = -2
GRAPH_Y_MAX = 2
GRAPH_FIGURE_SIZE = (5, 4)
GRAPH_DPI = 100
GRAPH_LINE_COLOR = "purple"
GRAPH_LINE_STYLE = "--"
GRAPH_POINT_COLORS = {
    (1, 1): "ro",
    (1, -1): "go",
    (-1, 1): "yo",
    (-1, -1): "bo",
}

# UI
WINDOW_GEOMETRY = "900x700"
WINDOW_TITLE = "PERCEPTRON SIMPLE - AND"
BUTTON_WIDTH = 15
FONT_TITLE = ("Helvetica", 10, "bold")
BUTTON_STYLE = "Action.TButton"
