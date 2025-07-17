import argparse
import json
import os

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import Dataset, load_dataset


def rouge_l(reference: str, prediction: str) -> float:
    """
    Compute ROUGE-L score based on longest common subsequence.

    Args:
        reference: Reference text
        prediction: Generated text

    Returns:
        ROUGE-L F1 score between 0 and 1
    """
    reference = reference.strip()
    prediction = prediction.strip()

    def lcs_length(x: str, y: str) -> int:
        # Build LCS matrix
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    lcs_len = lcs_length(reference, prediction)

    if len(reference) == 0 or len(prediction) == 0:
        return 0.0

    # Calculate precision and recall
    precision = lcs_len / len(prediction) if len(prediction) > 0 else 0
    recall = lcs_len / len(reference) if len(reference) > 0 else 0

    # Calculate F1 score
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)

    return f1


def exact_match(reference: str, prediction: str) -> bool:
    if 1.0 == rouge_l(reference, prediction):
        return True
    else:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompt_ratio", type=float, default=0.4)
    args = parser.parse_args()

    results_per_dataset = {}

    llm = LLM(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    sampling_params = SamplingParams(
        max_tokens=1024,
        temperature=0.0,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    for dataset in ["math500", "aime25", "aime24", "amc23", "gpqa", "livemathbench"]:
        # load the dataset
        if dataset == "math500":
            raw_dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        elif (
            dataset == "aime25"
            or dataset == "aime24"
            or dataset == "amc23"
            or dataset == "gpqa"
            or dataset == "livemathbench"
        ):
            # current file path
            dataset_path = f"{dataset}.jsonl"
            with open(dataset_path, "r") as f:
                raw_dataset = [json.loads(line) for line in f]
            raw_dataset = Dataset.from_list(raw_dataset)
        else:
            raise ValueError(f"Invalid dataset: {dataset}")

        # get all problem from the dataset.
        problems = [d["problem"].strip() for d in raw_dataset]
        problem_words = [p.split() for p in problems]
        _partial_problems = [
            " ".join(p[: int(len(p) * args.prompt_ratio)]).strip()
            for p in problem_words
        ]
        _ref_problems = [
            " ".join(p[int(len(p) * args.prompt_ratio) :]).strip()
            for p in problem_words
        ]
        partial_problems, ref_problems = [], []
        for p, r in zip(_partial_problems, _ref_problems):
            if len(p) > 0 and len(r) > 0:
                partial_problems.append(p)
                ref_problems.append(r)

        problem_predictions = llm.generate(
            partial_problems, sampling_params=sampling_params
        )
        problem_predictions = [p.outputs[0].text.strip() for p in problem_predictions]

        # now, let's compute the ROUGE-L and exact match scores.
        rouge_l_scores = [
            rouge_l(ref, pred[: len(ref)])
            for ref, pred in zip(ref_problems, problem_predictions)
        ]
        exact_match_scores = [
            exact_match(ref, pred[: len(ref)])
            for ref, pred in zip(ref_problems, problem_predictions)
        ]

        # print the results.
        print(f">>> {dataset}")
        print(f"ROUGE-L: {sum(rouge_l_scores) / len(rouge_l_scores):.2f}")
        print(f"Exact Match: {sum(exact_match_scores) / len(exact_match_scores):.2f}")

        results_per_dataset[dataset] = {
            "rouge_l_scores": rouge_l_scores,
            "exact_match_scores": exact_match_scores,
        }

        os.makedirs(f"../exp8", exist_ok=True)
        model_name = "_".join(args.model_path.split("/"))
        with open(
            f"../exp8/results_{dataset}_{args.prompt_ratio}_{model_name}.jsonl",
            "w",
        ) as f:
            for (
                rouge_l_score,
                exact_match_score,
                problem_prediction,
                partial_problem,
                ref_problem,
            ) in zip(
                rouge_l_scores,
                exact_match_scores,
                problem_predictions,
                partial_problems,
                ref_problems,
            ):
                f.write(
                    json.dumps(
                        {
                            "rouge_l_score": rouge_l_score,
                            "exact_match_score": exact_match_score,
                            "problem_prediction": problem_prediction[
                                : len(ref_problem)
                            ],
                            "partial_problem": partial_problem,
                            "ref_problem": ref_problem,
                        }
                    )
                    + "\n"
                )

    for dataset_name, results in results_per_dataset.items():
        print(f">>> {dataset_name}")
        print(
            f"ROUGE-L: {sum(results['rouge_l_scores']) / len(results['rouge_l_scores']):.2f}"
        )
        print(
            f"Exact Match: {sum(results['exact_match_scores']) / len(results['exact_match_scores']):.2f}"
        )
        print()

