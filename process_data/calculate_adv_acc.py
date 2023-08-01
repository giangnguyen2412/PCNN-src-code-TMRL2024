import numpy as np

filename = '../confidence.npy'
confidence_dict = np.load(filename, allow_pickle=True, ).item()

confidence_dict = dict(sorted(confidence_dict.items()))

def calculate_sum(data, confidence):
    total = 0
    for key, value in data.items():
        if key < confidence:
            total += value[-1]  # Add up the last element of the list
    return total

def calculate_accuracy(confidence_dict, key):
    correct_count = 0
    total_count = 0
    for confidence, counts in confidence_dict.items():
        if confidence >= key:
            correct_count += counts[0]
            # total_count += sum(counts)
            total_count += counts[0] + counts[1]

    # Now calculating the accuracy of thresholding
    correct_count += calculate_sum(confidence_dict, key)

    if total_count == 0:
        total_count = 1e6
    accuracy = correct_count*100/4794
    return accuracy

# Example usage
for confidence, counts in confidence_dict.items():
    accuracy = calculate_accuracy(confidence_dict, confidence)
    if confidence %5 == 0 or confidence == 99:
    # if True:
        print(f"Confidence scores: {confidence}, Advising network: {accuracy:.2f}")