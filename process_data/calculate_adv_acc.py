import numpy as np

filename = '../confidence.npy'
confidence_dict = np.load(filename, allow_pickle=True, ).item()

confidence_dict = dict(sorted(confidence_dict.items()))

def calculate_accuracy(confidence_dict, key):
    correct_count = 0
    total_count = 0
    for confidence, counts in confidence_dict.items():
        if confidence >= key:
            correct_count += counts[0]
            total_count += sum(counts)
    accuracy = correct_count*100/total_count
    return accuracy, total_count


# conf < T -> advising network | conf >= T -> thresholding
def calculate_accuracy_v2(confidence_dict, key):
    correct_count = 0
    total_count = 0
    for confidence, counts in confidence_dict.items():
        if confidence < key:
            correct_count += counts[0]
        else:
            correct_count += counts[2]

        total_count += counts[0] + counts[1]

    accuracy = correct_count*100/279
    return accuracy, total_count

# Example usage
for confidence, counts in confidence_dict.items():
    accuracy, total_count = calculate_accuracy_v2(confidence_dict, confidence)
    if confidence %5 == 0 or confidence == 99:
        # print(f"Confidence scores: >= {confidence}, Advising network: {accuracy:.2f}, % of samples: {total_count*100/279:.2f}")
        print(f"Confidence scores: >= {confidence}, Advising network: {accuracy:.2f}")