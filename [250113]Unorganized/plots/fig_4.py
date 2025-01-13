import matplotlib.pyplot as plt

sample_num = ['None', 'Top 1', 'Top 2', 'Top 3']
avg_token_num = [1000, 2000, 2500, 3000]
general_perception = [45, 48, 50, 47]
driving_suggestion = [50, 51, 53, 52]

plt.rcParams['font.family'] = 'Times New Roman'

fig, ax1 = plt.subplots(figsize=(10, 6))

color = '#4682B4'
bars = ax1.bar(sample_num, avg_token_num, label="Avg Token Num", color=color, alpha=0.7, width=0.4)

for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), va='bottom', ha='center', fontsize=10)

ax1.set_xlabel("Sample Num", fontsize=15)
ax1.set_ylabel("Avg Token Num", fontsize=15, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

ax2.plot(sample_num, general_perception, label="General Perception", linestyle='--', marker='s', color='#2E8B57', linewidth=2.5, markersize=8)
ax2.plot(sample_num, driving_suggestion, label="Driving Suggestion", linestyle='-', marker='^', color='#DC143C', linewidth=2.5, markersize=8)

ax2.set_ylabel("Scores", fontsize=15)
ax2.tick_params(axis='y')

fig.suptitle("Performance Analysis vs. Sample Num", fontsize=18, weight='bold')
ax1.set_title("Figure 1: Avg Token Num and Scores for General Perception and Driving Suggestion", fontsize=12, pad=20)

ax1.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.3)
ax1.grid(False, axis='x')

ax1_legend = ax1.legend(loc='upper left', fontsize=12, frameon=True, framealpha=0.8, shadow=True)
ax2_legend = ax2.legend(loc='upper right', fontsize=12, frameon=True, framealpha=0.8, shadow=True)

plt.tight_layout()
plt.savefig("/mnt/data/cvpr_style_plot.pdf", format='pdf', dpi=300)
plt.show()
