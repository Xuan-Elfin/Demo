import pandas as pd
import seaborn as sns

# Visualize the accuracy of the classifier for different percentages of training labels
accuracy_results_df = pd.DataFrame(accuracy_results)

f, ax = plt.subplots(1,1)

sns.lineplot(data=accuracy_results_df, x='Number of Labels', y='Test Accuracy', hue='Model', ax=ax)
ax.set_xscale('log')
