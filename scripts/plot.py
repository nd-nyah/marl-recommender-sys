import os
import json
import argparse
import matplotlib.pyplot as plt


def load_results(path):

    with open(path, "r") as f:
        return json.load(f)


def plot_metrics(results, save_dir):

    os.makedirs(save_dir, exist_ok=True)

    metrics = results[0].keys()

    for metric in metrics:

        values = [r[metric] for r in results]

        plt.figure()
        plt.plot(values, marker="o")
        plt.title(metric)
        plt.xlabel("Evaluation Run")
        plt.ylabel(metric)

        plt.grid(True)

        plot_path = os.path.join(save_dir, f"{metric}.png")
        plt.savefig(plot_path)

        plt.close()

        print(f"Saved plot → {plot_path}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to JSON evaluation results"
    )

    parser.add_argument(
        "--out",
        type=str,
        default="results/evaluate/plots",
        help="Output directory for plots"
    )

    args = parser.parse_args()

    results = load_results(args.input)

    plot_metrics(results, args.out)


if __name__ == "__main__":
    main()