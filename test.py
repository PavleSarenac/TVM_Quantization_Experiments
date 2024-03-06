import numpy as np
import matplotlib.pyplot as plt

labels = [
    "Unquantized",
    "Int8 Quantized(activation=int32)",
    "Int8 Quantized(activation=int16)",
    "Int8 Quantized(activation=int8)"
]

average_inference_times = np.array([[
    0.3,
    0.6,
    0.5,
    0.8
]])

for i in range(4):
    plt.bar(labels[i], average_inference_times[:, i], width=0.3)

note_text = "Hardware device that was used: Intel(R) Core(TM) i7-10510U CPU"
plt.legend(loc="upper left")
plt.annotate(note_text, xy=(0.5, -0.1), xycoords="axes fraction", ha="center", fontsize=8)

plt.title("ResNet50 Machine Learning Model Compiled With TVM")
plt.ylabel("Average Inference Time (seconds)")

plt.show()
