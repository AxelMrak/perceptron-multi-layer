import tkinter as tk

from config import LEARNING_RATE
from model.perceptron import Perceptron
from model.activation import StepActivation
from model.dataset import LogicGateDataset
from controller.trainer import TrainerController
from view.tkinter_view import TkinterView
from view.graph_view import GraphView


def main() -> None:
    window = tk.Tk()

    view = TkinterView(window)
    graph = GraphView(window)

    #    3 inputs: bias (always 1.0) + x1 + x2
    #    Learning rate from config.py (only main.py touches config)
    perceptron = Perceptron(input_size=3, activation=StepActivation(), learning_rate=LEARNING_RATE)

    dataset = LogicGateDataset.and_gate()
    controller = TrainerController(perceptron, dataset)


    def on_train() -> None:
        """Full training: run until convergence."""
        graph.clear()
        converged = controller.train_full()
        weights = controller.perceptron.weights
        view.update_training_display(
            epoch=controller.current_epoch,
            converged=converged,
            sample_input1="—",
            sample_input2="—",
            weight1=f"{weights[1]:.4f}",
            weight2=f"{weights[2]:.4f}",
            bias_weight=f"{weights[0]:.4f}",
            desired="—",
            obtained="—",
        )
        graph.plot_decision_boundary(
            bias_weight=weights[0],
            weight1=weights[1],
            weight2=weights[2],
            epoch=controller.current_epoch,
        )

    def on_learn_step() -> None:
        """One epoch at a time ."""
        converged = controller.train_epoch()
        weights = controller.perceptron.weights
        sample = controller.current_sample_input
        view.update_training_display(
            epoch=controller.current_epoch,
            converged=converged,
            sample_input1=f"{sample[1]:.0f}",
            sample_input2=f"{sample[2]:.0f}",
            weight1=f"{weights[1]:.4f}",
            weight2=f"{weights[2]:.4f}",
            bias_weight=f"{weights[0]:.4f}",
            desired=f"{controller.current_sample_expected:.0f}",
            obtained=f"{controller.current_sample_predicted:.0f}",
        )
        graph.plot_decision_boundary(
            bias_weight=weights[0],
            weight1=weights[1],
            weight2=weights[2],
            epoch=controller.current_epoch,
        )
        if controller.current_error != 0.0:
            view.show_info(
                f"Weights recalculated\n"
                f"New bias weight: {weights[0]:.4f}\n"
                f"New weight 1: {weights[1]:.4f}\n"
                f"New weight 2: {weights[2]:.4f}"
            )

    def on_test() -> None:
        """Test the perceptron with user-provided inputs."""
        try:
            result = controller.test_prediction(
                view.test_input1.get(),
                view.test_input2.get(),
            )
            view.update_test_result(result)
        except ValueError:
            view.show_error("Only values 1 or -1 are accepted.")

    view.train_button.config(command=on_train)
    view.learn_button.config(command=on_learn_step)
    view.test_button.config(command=on_test)

    window.mainloop()


if __name__ == "__main__":
    main()
