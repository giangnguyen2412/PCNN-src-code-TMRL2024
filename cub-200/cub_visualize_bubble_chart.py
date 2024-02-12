import matplotlib.scale as mscale
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.scale import ScaleBase
from matplotlib.transforms import Transform
from matplotlib.ticker import AutoLocator, ScalarFormatter
import pandas as pd
import matplotlib.lines as mlines

class CustomScale(ScaleBase):
    name = "custom"

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
            return np.where(
                values <= self.threshold,
                values,
                self.threshold
                + (self.max_range - self.threshold)
                * (values - self.threshold)
                / (self.max_range - self.threshold),
            )

        def inverted(self):
            return CustomScale.InvertedCustomTransform(self.threshold, self.max_range)

    class InvertedCustomTransform(Transform):
        input_dims = output_dims = 1

        def __init__(self, threshold, max_range):
            super().__init__()
            self.threshold = threshold
            self.max_range = max_range

        def transform_non_affine(self, values):
            return np.where(
                values <= self.threshold,
                values,
                self.threshold
                + (values - self.threshold)
                * (self.max_range - self.threshold)
                / (self.max_range - self.threshold),
            )

        def inverted(self):
            return CustomScale.CustomTransform(self.threshold, self.max_range)


# Register the custom scale with matplotlib.scale, not matplotlib.pyplot
mscale.register_scale(CustomScale)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the Excel file
data = pd.read_excel("data.xlsx", skiprows=5)  # Adjust skiprows as needed

# Clean the data by selecting relevant columns and renaming them for clarity
cleaned_data = data[["Model", "Params", "Acc"]].copy()

# Convert 'Params' column to numeric for plotting
cleaned_data["Params"] = pd.to_numeric(cleaned_data["Params"], errors="coerce")


def adjust_params(row):
    # Define the milestones and their corresponding scaling targets
    milestones = [
        (12e6, 12e6),
        (20e6, 50e6),
        (30e6, 90e6),
        (100e6, 130e6),
        (120e6, 160e6),
        (180e6, 180e6),
    ]

    # Get the current number of parameters
    current_params = row["Params"]

    # Determine the closest milestone
    closest_milestone, target_scale = min(
        milestones, key=lambda x: abs(current_params - x[0])
    )

    # Calculate the new number of parameters based on the scaling formula
    new_params = (current_params / closest_milestone) * target_scale

    return new_params


# Apply the adjust_params function to each row in the DataFrame
cleaned_data["Params"] = cleaned_data.apply(adjust_params, axis=1)

# Assign colors based on explicit backbone definitions
explicit_backbones = cleaned_data["Model"].apply(
    lambda x: x.replace("xS", "") if "xS" in x else x
)
backbone_explicit_colors = {
    backbone: plt.cm.tab20(i) for i, backbone in enumerate(explicit_backbones.unique())
}


contrasting_colors = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # lime
    "#17becf",  # cyan
]

# Map each explicit backbone to a contrasting color
unique_backbones = (
    cleaned_data["Model"]
    .apply(lambda x: x.replace("xS", "") if "xS" in x else x)
    .unique()
)
backbone_to_color = {
    backbone: contrasting_colors[i % len(contrasting_colors)]
    for i, backbone in enumerate(unique_backbones)
}

# Adjusting the plot with the new x-axis scale
fig, ax = plt.subplots(figsize=(10, 5.5))

# Set the custom scale for the x-axis
ax.set_xscale("custom", threshold=5e6, max_range=195e6)

for _, row in cleaned_data.iterrows():
    model_backbone_explicit = (
        row["Model"].replace("xS", "") if "xS" in row["Model"] else row["Model"]
    )
    if "xS" in row["Model"]:
        marker = "*"  # Triangle marker for larger models
    else:
        marker = "o"  # Circle marker for smaller models
    ax.scatter(
        row["Params"],
        row["Acc"],
        s=100,
        c=[backbone_to_color[model_backbone_explicit]],
        alpha=0.6,
        marker=marker,
        label=model_backbone_explicit,
    )  # Set the size to 100 for uniformity

# Enable gridlines for both x and y axes
ax.grid(True, linestyle="-", linewidth=0.5, color="gray", alpha=0.5)

# Set the axis labels with increased font size
ax.set_xlabel("Number of Parameters", fontsize=16)
ax.set_ylabel("CUB-200 Top-1 Accuracy (%)", fontsize=16)

ax.set_xlim(5e6, 195e6)  # Extend x-axis range from 10M to 190M

# Customize x-axis ticks to specific values (in actual data units, which might need adjustment)
x_tick_values = [12e6, 50e6, 90e6, 130e6, 160e6, 180e6]
ax.set_xticks(x_tick_values)
x_tick_labels = ["12M", "20M", "30M", "100M", "120M", "180M"]
ax.set_xticklabels(
    x_tick_labels, fontsize=14
)  # Adjust font size for x-axis tick labels

# Remove top and right borders
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Make left and bottom borders thicker
ax.spines["left"].set_linewidth(5)
ax.spines["bottom"].set_linewidth(5)


# set xlim
ax.set_xlim(5e6, 181e6)

# Update the arrow drawing section to change the color to grey
for model in cleaned_data["Model"].unique():
    if "xS" in model:
        base_model = model.replace("xS", "")
        if base_model in cleaned_data["Model"].values:
            base_data = cleaned_data[cleaned_data["Model"] == base_model].iloc[0]
            extended_data = cleaned_data[cleaned_data["Model"] == model].iloc[0]
            # Check if the model is 'INat-RN50'
            if "INat-RN50" in model:
                arrow_style = "-"  # Solid arrow for 'INat-RN50'
                linestyle = "solid"
                arrow_color = "grey"  # Change arrow color to grey
            else:
                arrow_style = "->"  # Arrow style remains the same
                linestyle = "dashed"  # More dashed line for other models
                arrow_color = "grey"  # Change arrow color to grey
            ax.annotate(
                "",
                xy=(extended_data["Params"], extended_data["Acc"]),
                xytext=(base_data["Params"], base_data["Acc"]),
                arrowprops=dict(
                    arrowstyle=arrow_style, linestyle=linestyle, color=arrow_color, lw=1
                ),
            )


# Adjust the legend to use square markers for all items and reflect the colors in the plots
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))

# Create custom legend entries with square markers for all items, using the colors from the plot
custom_handles = [
    mlines.Line2D(
        [],
        [],
        color=backbone_to_color[label],
        marker="s",
        linestyle="None",
        markersize=10,
        label=label,
    )
    for label in by_label.keys()
]

# Include custom entries for 'Seen' and 'Unseen' with square markers, using specific colors
custom_handles += [
    mlines.Line2D([], [], color="grey", marker="s", linestyle="-", lw=1, label="Seen"),
    mlines.Line2D(
        [], [], color="grey", marker="s", linestyle="dashed", lw=1, label="Unseen"
    ),
]

# Update the legend with the new custom handles
legend = ax.legend(
    handles=custom_handles, loc="upper left", fontsize="x-large", title=""
)

for legend_handle in legend.legendHandles:
    legend_handle._sizes = [100]

fig.savefig("bubble_plot.pdf", dpi=300, bbox_inches="tight")
plt.show()