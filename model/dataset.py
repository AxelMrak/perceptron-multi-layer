import csv
from dataclasses import dataclass
from config import AND_INPUTS, AND_OUTPUTS


@dataclass
class TrainingSample:
    """A single training example with inputs and expected output."""
    inputs: list[float]
    expected: float


class Dataset:
    """Iterable collection of TrainingSamples."""

    def __init__(self, samples: list[TrainingSample]) -> None:
        self._samples = samples

    def __iter__(self):
        return iter(self._samples)

    def __len__(self) -> int:
        return len(self._samples)


class FootballDataset:
    """Factory for football match datasets from CSV."""

    @staticmethod
    def from_csv(filepath: str, normalize: bool = True) -> Dataset:
        """Load football data from CSV. Last column is the expected output."""
        rows = []
        with open(filepath, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].startswith("#"):
                    continue
                values = [float(v) for v in row]
                rows.append((values[:-1], values[-1]))

        if not rows:
            raise ValueError(f"No data found in {filepath}")

        if normalize:
            num_features = len(rows[0][0])
            for col in range(num_features):
                col_values = [r[0][col] for r in rows]
                cmin, cmax = min(col_values), max(col_values)
                if cmax > cmin:
                    for r in rows:
                        r[0][col] = 2.0 * (r[0][col] - cmin) / (cmax - cmin) - 1.0

        samples = [
            TrainingSample(inputs=[1.0] + features, expected=expected)
            for features, expected in rows
        ]
        return Dataset(samples)


class LogicGateDataset:
    """Factory for logic gate datasets."""

    @staticmethod
    def and_gate() -> Dataset:
        """Build the AND gate training dataset from config."""
        samples = [
            TrainingSample(inputs=inp, expected=exp)
            for inp, exp in zip(AND_INPUTS, AND_OUTPUTS)
        ]
        return Dataset(samples)
