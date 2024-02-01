import matplotlib.scale as mscale
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.scale import ScaleBase
from matplotlib.transforms import Transform
from matplotlib.ticker import AutoLocator, ScalarFormatter
import pandas as pd

class CustomScale(ScaleBase):
    name = 'custom'

    def __init__(self, axis, *, threshold=100e6, max_range=180e6, **kwargs):
        super().__init__(axis)
        self.threshold = threshold
        self.max_range = max_range

    def get_transform(self):
        return self.CustomTransform(self.threshold, self.max_range)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(AutoLocator())
        axis.set_major_formatter(ScalarFormatter())

    class CustomTransform(Transform):
        input_dims = output_dims = 1

        def __init__(self, threshold, max_range):
            super().__init__()
            self.threshold = threshold
            self.max_range = max_range

        def transform_non_affine(self, values):
            return np.where(values <= self.threshold,
                            values,
                            self.threshold + (self.max_range - self.threshold) * (values - self.threshold) / (self.max_range - self.threshold))

        def inverted(self):
            return CustomScale.InvertedCustomTransform(self.threshold, self.max_range)

    class InvertedCustomTransform(Transform):
        input_dims = output_dims = 1

        def __init__(self, threshold, max_range):
            super().__init__()
            self.threshold = threshold
            self.max_range = max_range

        def transform_non_affine(self, values):
            return np.where(values <= self.threshold,
                            values,
                            self.threshold + (values - self.threshold) * (self.max_range - self.threshold) / (self.max_range - self.threshold))

        def inverted(self):
            return CustomScale.CustomTransform(self.threshold, self.max_range)

# Register the custom scale with matplotlib.scale, not matplotlib.pyplot
mscale.register_scale(CustomScale)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the Excel file
data = pd.read_excel('data.xlsx', skiprows=5)  # Adjust skiprows as needed

# Clean the data by selecting relevant columns and renaming them for clarity
cleaned_data = data[['Model', 'Params', 'Acc']].copy()

# Convert 'Params' column to numeric for plotting
cleaned_data['Params'] = pd.to_numeric(cleaned_data['Params'], errors='coerce')


def adjust_params(row):
    # Define the milestones and their corresponding scaling targets
    milestones = [(12e6, 12e6), (20e6, 50e6), (30e6, 90e6),
                  (100e6, 130e6), (120e6, 160e6), (180e6, 180e6)]

    # Get the current number of parameters
    current_params = row['Params']

    # Determine the closest milestone
    closest_milestone, target_scale = min(milestones, key=lambda x: abs(current_params - x[0]))

    # Calculate the new number of parameters based on the scaling formula
    new_params = (current_params / closest_milestone) * target_scale

    return new_params


def adjust_acc(row):
    # Define the milestones and their corresponding scaling targets for accuracies
    # These are hypothetical and should be adjusted based on your data and requirements
    milestones = [(60, 60), (80, 70), (85, 80), (90, 90)]

    # Get the current accuracy
    current_acc = row['Acc']

    # Determine the closest milestone
    closest_milestone, target_scale = min(milestones, key=lambda x: abs(current_acc - x[0]))

    # Calculate the new accuracy based on the scaling formula
    # This is a simple linear transformation for demonstration purposes
    new_acc = (current_acc - closest_milestone) / (100 - closest_milestone) * (target_scale - closest_milestone) + closest_milestone

    return new_acc

# Apply the adjust_acc function to each row in the DataFrame for the 'Acc' column
cleaned_data['Acc'] = cleaned_data.apply(adjust_acc, axis=1)

# Apply the adjust_params function to each row in the DataFrame
cleaned_data['Params'] = cleaned_data.apply(adjust_params, axis=1)

# Assign colors based on explicit backbone definitions
explicit_backbones = cleaned_data['Model'].apply(lambda x: x.replace('xS', '') if 'xS' in x else x)
backbone_explicit_colors = {backbone: plt.cm.tab20(i) for i, backbone in enumerate(explicit_backbones.unique())}


contrasting_colors = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # lime
    '#17becf'   # cyan
]

# Map each explicit backbone to a contrasting color
unique_backbones = cleaned_data['Model'].apply(lambda x: x.replace('xS', '') if 'xS' in x else x).unique()
backbone_to_color = {backbone: contrasting_colors[i % len(contrasting_colors)] for i, backbone in enumerate(unique_backbones)}


# Adjusting the plot with the new x-axis scale
fig, ax = plt.subplots(figsize=(10, 8))

# Set the custom scale for the x-axis
ax.set_xscale('custom', threshold=5e6, max_range=195e6)

# # Plotting with the adjusted x-axis
# for _, row in cleaned_data.iterrows():
#     model_backbone_explicit = row['Model'].replace('xS', '') if 'xS' in row['Model'] else row['Model']
#     ax.scatter(row['Params'], row['Acc'], s=(row['Params'] / 6e4)/2, c=[backbone_to_color[model_backbone_explicit]], alpha=0.6, marker='o', label=model_backbone_explicit)

for _, row in cleaned_data.iterrows():
    model_backbone_explicit = row['Model'].replace('xS', '') if 'xS' in row['Model'] else row['Model']
    # Check if the model is a larger variant
    if 'xS' in row['Model']:  # Assuming 'large' is used to denote larger models
        marker = '*'  # Triangle marker for larger models
    else:
        marker = 'o'  # Circle marker for smaller models
    ax.scatter(row['Params'], row['Acc'], s=(row['Params'] / 6e4)/2, c=[backbone_to_color[model_backbone_explicit]], alpha=0.6, marker=marker, label=model_backbone_explicit)


# Manually set x-axis ticks at desired uniform intervals
x_ticks = np.linspace(5e6, 195e6, num=8)  # Adjust 'num' for the number of ticks you want
ax.set_xticks(x_ticks)

# y_ticks = np.linspace(60, 90)  # Adjust 'num' for the number of ticks you want
# ax.set_yticks(y_ticks)

# Enable gridlines only for the x-axis
# ax.grid(True, axis='x', linestyle='-', linewidth=0.5, color='gray', alpha=0.5)
# ax.grid(True, axis='y', linestyle='-', linewidth=0.5, color='gray', alpha=0.5)

# # Add dashed-thin arrows for connections
# for model in cleaned_data['Model'].unique():
#     if 'xS' in model:
#         base_model = model.replace('xS', '')
#         if base_model in cleaned_data['Model'].values:
#             base_data = cleaned_data[cleaned_data['Model'] == base_model].iloc[0]
#             extended_data = cleaned_data[cleaned_data['Model'] == model].iloc[0]
#             ax.annotate('', xy=(extended_data['Params'], extended_data['Acc']),
#                         xytext=(base_data['Params'], base_data['Acc']),
#                         arrowprops=dict(arrowstyle='->', linestyle='dashed', color='grey', lw=1))


# Update the arrow drawing section
for model in cleaned_data['Model'].unique():
    if 'xS' in model:
        base_model = model.replace('xS', '')
        if base_model in cleaned_data['Model'].values:
            base_data = cleaned_data[cleaned_data['Model'] == base_model].iloc[0]
            extended_data = cleaned_data[cleaned_data['Model'] == model].iloc[0]
            # Check if the model is 'INat-RN50'
            if 'INat-RN50' in model:
                arrow_style = '->'  # Solid arrow for 'INat-RN50'
                linestyle = 'solid'
            else:
                arrow_style = '->'  # Arrow style remains the same
                linestyle = 'dashed'  # More dashed line for other models
            ax.annotate('', xy=(extended_data['Params'], extended_data['Acc']),
                        xytext=(base_data['Params'], base_data['Acc']),
                        arrowprops=dict(arrowstyle=arrow_style, linestyle=linestyle, color='grey', lw=1))

# Set the axis labels with increased font size
ax.set_xlabel('Number of Parameters', fontsize=16)  # Increase font size as needed
ax.set_ylabel('CUB-200 Top-1 Accuracy (%)', fontsize=16)  # Increase font size as needed

# ax.set_title('Bubble Plot with Adjusted X-Axis Scale')
# ax.grid(False)
ax.set_xlim(5e6, 195e6)  # Extend x-axis range from 10M to 190M


# Customize x-axis ticks to specific values (in actual data units, which might need adjustment)
x_tick_values = [12e6, 50e6, 90e6, 130e6, 160e6, 180e6]
# Set the x-axis ticks
ax.set_xticks(x_tick_values)

# Set the x-axis tick labels to desired text labels
x_tick_labels = ['12M', '20M', '30M', '100M', '120M', '180M']
ax.set_xticklabels(x_tick_labels)

# Customize x-axis ticks to specific values (in actual data units, which might need adjustment)
y_tick_values = [60, 70, 80, 90]
# Set the x-axis ticks
ax.set_yticks(y_tick_values)

# Set the x-axis tick labels to desired text labels
y_tick_labels = ['60', '80', '85', '90']
ax.set_yticklabels(y_tick_labels)



# Adjust the legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
legend = ax.legend(by_label.values(), by_label.keys(), title='', loc='lower right', fontsize='x-large')  # Increase fontsize here
for legend_handle in legend.legendHandles:
    legend_handle._sizes = [100]  # Adjust bubble size in the legend to be uniform


import matplotlib.lines as mlines

# Create a list of custom legend handles with square markers for each unique model
custom_handles = []
for model, color in backbone_to_color.items():
    handle = mlines.Line2D([], [], color=color, marker='s', linestyle='None',
                           markersize=10, label=model)
    custom_handles.append(handle)

# Create the main legend with custom square markers
main_legend = ax.legend(handles=custom_handles, loc='lower right', fontsize='x-large', title='')
ax.add_artist(main_legend)  # Add the main legend back as an artist


# Create custom legend entries for "Seen" and "Unseen"
seen_marker = mlines.Line2D([], [], color='grey', linestyle='solid', lw=2, label='Seen')
unseen_marker = mlines.Line2D([], [], color='grey', linestyle='dashed', lw=2, label='Unseen')

# Update the legend creation section
# Reuse the 'custom_handles' list from the main legend creation
# Add the new legend entries for "Seen" and "Unseen"
custom_handles.extend([seen_marker, unseen_marker])

# Update the main legend to include the new entries
main_legend = ax.legend(handles=custom_handles, loc='upper left', fontsize='x-small', title='')  # Adjust font size and title as needed

# Add the updated main legend back to the plot
ax.add_artist(main_legend)

# Create the second legend for the classifier type (C and CxS)
circle_marker = mlines.Line2D([], [], color='grey', marker='o', linestyle='None',
                              markersize=15, label='$\\mathbf{C}$')  # Using LaTeX for bold "C"
star_marker = mlines.Line2D([], [], color='grey', marker='*', linestyle='None',
                            markersize=20, label='$\\mathbf{C\\times S}$')  # Using LaTeX for bold "CxS"

# Create and add the second legend to the plot
second_legend = ax.legend(handles=[circle_marker, star_marker], loc='upper left', fontsize='x-large')

# Add the second legend as an artist to ensure it doesn't remove the main legend
ax.add_artist(second_legend)

# Tight layout to minimize redundant spaces
plt.tight_layout()

# Save the plot as a PDF file
# plt.savefig('bubles.pdf', format='pdf', bbox_inches='tight')

plt.show()
