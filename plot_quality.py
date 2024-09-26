import ast
import csv

import matplotlib.pyplot as plt

groups = {
    "256": {"max_y": 0, "plots": []},
}
with open("quality.csv", mode="r", newline="") as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        key = row[1]
        label = f"{row[0]},{row[1]},{row[2]},{row[3]}"
        rmse = ast.literal_eval(row[4])
        x = list(range(len(rmse) - 2))
        groups[key]["max_y"] = max(groups[key]["max_y"], max(rmse[2:30]))
        groups[key]["plots"].append({"x": x, "y": rmse[2:], "label": label})

# plt.figure(figsize=(8, 32))
for i, key in enumerate(groups):
    plt.subplot(len(groups), 1, i + 1)
    ax = plt.gca()
    ax.set_ylim([0, groups[key]["max_y"]])
    for plot in groups[key]["plots"]:
        linestyle="-"
        plt.plot(plot["x"], plot["y"], label=plot["label"], linestyle=linestyle)

    plt.title(key)

# Display the plots
plt.tight_layout()  # Adjust the spacing
plt.legend()
plt.show()
