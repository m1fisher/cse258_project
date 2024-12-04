# c/o gpt4 for initial code
from collections import defaultdict
import re
import statistics
# Define the input file name
input_file = "TwoStage_eval_stats.txt"

# Initialize variables to hold cumulative sums and count of CSV files
total_recall = 0
total_r_precision = 0
total_num_clicks = 0
total_ndcg = 0
file_count = 0

def _format(y):
    return f"{y:.3f}"

# Open and process the input file
with open(input_file, "r") as file:
    lines = file.readlines()
    r_precisions = defaultdict(list)
    for i in range(len(lines)):
        # Check if the line contains a CSV file name
        if lines[i].strip().endswith(".csv"):
            file_count += 1  # Increment file count
            match = re.search(r'\d+', lines[i])
            if match:
                num = int(match.group())
        # Extract metrics from subsequent lines
        elif "Average recall" in lines[i]:
            total_recall += float(lines[i].split(":")[1].strip())
        elif "Average R-precision" in lines[i]:
            total_r_precision += float(lines[i].split(":")[1].strip())
            r_precisions[num].append(float(lines[i].split(":")[1].strip()))
        elif "Average num clicks" in lines[i]:
            total_num_clicks += float(lines[i].split(":")[1].strip())
        elif "Average NDCG" in lines[i]:
            total_ndcg += float(lines[i].split(":")[1].strip())
    print({k: statistics.mean(v) for k, v in r_precisions.items()})

# Calculate averages
average_recall = total_recall / file_count
average_r_precision = total_r_precision / file_count
average_num_clicks = total_num_clicks / file_count
average_ndcg = total_ndcg / file_count

# Print the results
print("Averaged Metrics Across All CSV Files:")
print(f"Average Recall: {average_recall:.3f}")
print(f"Average R-Precision: {average_r_precision:.3f}")
print(f"Average Number of Clicks: {average_num_clicks:.3f}")
print(f"Average NDCG: {average_ndcg:.3f}")

