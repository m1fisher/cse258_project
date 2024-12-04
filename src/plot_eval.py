import matplotlib.pyplot as plt

metrics = {
    "TwoStage Model": {1: 0.08724997619661899, 5: 0.13757730934346155, 10: 0.15300822174628115, 25: 0.1991121163310166, 100: 0.13859424471774173},
    "LatentFactors": {1: 0.08454495135604849, 5: 0.11787639295996868, 10: 0.1249086476653726, 25: 0.14758059468601503, 100: 0.08747230272153012},
    "PopularityBaseline": {1: 0.018571943173355712, 5: 0.022883774388527636, 10: 0.03308525311872319, 25: 0.043659670095345104, 100: 0.03621250872693846},
}

# c/o gpt4 for initial code
# Create the plot
plt.figure(figsize=(20, 12))

# Iterate through each row and plot the values
for label, data in metrics.items():
    x = list(data.keys())
    y = list(data.values())
    plt.plot(x, y, 'o-', label=label)  # Use dots for data points

# Customize tick label font size
plt.tick_params(axis='both', which='major', labelsize=12)  # Increase font size for major ticks
plt.tick_params(axis='both', which='minor', labelsize=10)  # Increase font size for minor ticks

plt.xscale('log')
# Customize the plot
plt.title("Model Performance Relative to Input Length", fontsize=24)
plt.xlabel("Num. input tracks", fontsize=24)
plt.ylabel("R-precision", fontsize=24)
plt.legend(title="Metrics", fontsize=16)
plt.grid(True, linestyle="--", alpha=0.7)


plt.savefig("metrics_plot.png", dpi=300, bbox_inches='tight')  # Save as a PNG file with high resolution
# Show the plot
#plt.show()
