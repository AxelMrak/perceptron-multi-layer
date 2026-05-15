from model.network import Network
from model.layer import Layer
from model.activation import SigmoidActivation, TanhActivation
from model.dataset import FootballDataset
from controller.trainer import TrainerController


def main() -> None:
    LEARNING_RATE = 0.1
    INPUT_SIZE = 7       # bias + 6 features
    HIDDEN_NEURONS = 4
    OUTPUT_INPUTS = 5    # bias + 4 hidden outputs

    hidden_layer = Layer(
        num_neurons=HIDDEN_NEURONS,
        num_inputs=INPUT_SIZE,
        activation=SigmoidActivation(),
        learning_rate=LEARNING_RATE,
    )

    output_layer = Layer(
        num_neurons=1,
        num_inputs=OUTPUT_INPUTS,
        activation=TanhActivation(),
        learning_rate=LEARNING_RATE,
    )

    network = Network([hidden_layer, output_layer])

    dataset = FootballDataset.from_csv("data/football.csv")

    controller = TrainerController.__new__(TrainerController)
    converged, epochs = controller.train_backprop(network, dataset, max_epochs=5000)

    print(f"Training {'converged' if converged else 'did not converge'} after {epochs} epochs")
    print(f"Final MSE: {controller._current_error:.6f}")


if __name__ == "__main__":
    main()
