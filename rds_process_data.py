import json
import requests
import os


def compute_accuracies_from_jsonl(file_path):
    # Load data from the JSONL file
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))

    # Confidence levels
    confidence_levels = set([entry['self-reported-confidence'] for entry in data])

    # Initialize counters for overall, 'Accept', and 'Reject' accuracies
    correct_decisions = 0
    correct_accept_decisions = 0
    correct_reject_decisions = 0
    total_accept_decisions = 0
    total_reject_decisions = 0

    # Initialize variables for accuracy per confidence level
    correct_per_confidence = {}
    total_per_confidence = {}

    # Loop through each entry in the data
    for entry in data:
        gt_label = entry.get('gt-label', '').lower().replace(' ', '_').replace('-', '_')
        ai_prediction_label = entry.get('ai-prediction', {}).get('label', '').split(' ')[-1].lower().replace(' ',
                                                                                                             '_').replace(
            '-', '_')

        # Check if the ground truth and AI prediction match
        expected_decision = 'Accept' if gt_label == ai_prediction_label else 'Reject'

        # Update overall and specific correct decisions count
        if entry['decision'] == expected_decision:
            correct_decisions += 1
            if expected_decision == 'Accept':
                correct_accept_decisions += 1
            else:
                correct_reject_decisions += 1

        # Update total counts for 'Accept' and 'Reject' expected decisions
        if expected_decision == 'Accept':
            total_accept_decisions += 1
        else:
            total_reject_decisions += 1

        # Update correct and total counts for this confidence level
        confidence_level = entry['self-reported-confidence']
        if confidence_level not in correct_per_confidence:
            correct_per_confidence[confidence_level] = 0
            total_per_confidence[confidence_level] = 0
        if entry['decision'] == expected_decision:
            correct_per_confidence[confidence_level] += 1
        total_per_confidence[confidence_level] += 1

    # Compute and display accuracies
    overall_accuracy = correct_decisions / len(data) if data else 0
    accept_accuracy = correct_accept_decisions / total_accept_decisions if total_accept_decisions else 0
    reject_accuracy = correct_reject_decisions / total_reject_decisions if total_reject_decisions else 0

    print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")
    print(f"Accuracy when Expected Decision is 'Accept': {accept_accuracy * 100:.2f}% (Correct: {correct_accept_decisions}, Total: {total_accept_decisions})")
    print(f"Accuracy when Expected Decision is 'Reject': {reject_accuracy * 100:.2f}% (Correct: {correct_reject_decisions}, Total: {total_reject_decisions})")
    print("\nAccuracy per Confidence Level:")
    for level in confidence_levels:
        accuracy_per_conf = correct_per_confidence[level] / total_per_confidence[level] if total_per_confidence[
            level] else 0
        print(
            f"{level}: {accuracy_per_conf * 100:.2f}% (Correct: {correct_per_confidence[level]}, Total: {total_per_confidence[level]})")

# Example usage
file_path = 'son-87jl3nq4-DEBUG_decisions.jsonl'  # Replace with your file path
print(file_path)
base_url = 'https://huggingface.co/datasets/luulinh90s/chm-corr-prj-giang/raw/main'
file_path = os.path.join(base_url, file_path)
response = requests.get(file_path)
base = os.path.basename(file_path)
file_path = f"jsonls/{os.path.basename(base)}"

# Check if the request was successful
if response.status_code == 200:
    with open(file_path, "wb") as f:
        f.write(response.content)
    print("File downloaded successfully.")
else:
    print(f"Failed to download the file. Status code: {response.status_code}")

compute_accuracies_from_jsonl(file_path)

