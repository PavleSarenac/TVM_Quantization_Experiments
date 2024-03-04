########################################################################################################################
# PyTorch imports
# ----------------------------------------------------------------------------------------------------------------------
import torch
import torchvision
from torchvision import transforms
########################################################################################################################


########################################################################################################################
# TVM imports
# ----------------------------------------------------------------------------------------------------------------------
import tvm
from tvm import relay
from tvm.contrib.download import download_testdata
from tvm.contrib import graph_executor
########################################################################################################################


########################################################################################################################
# Other imports
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

########################################################################################################################


########################################################################################################################
# Getting an instance of a pretrained PyTorch ResNet50 model in inference mode
# ----------------------------------------------------------------------------------------------------------------------
model_name = "resnet50"
model = getattr(torchvision.models, model_name)(pretrained=True)
model = model.eval()
########################################################################################################################


########################################################################################################################
# Getting a TorchScripted representation of the ResNet50 model so that it can later be converted to a Relay graph
# ----------------------------------------------------------------------------------------------------------------------
# Obviously, we will be getting this TorchScripted model also in inference mode
input_shape = [1, 3, 224, 224]  # Input for our ResNet50 model will be a 4D tensor with NCHW data layout
input_data = torch.randn(input_shape)  # Randomized dummy input for the purpose of getting a TorchScripted model
torch_scripted_model = torch.jit.trace(model, input_data).eval()
########################################################################################################################


########################################################################################################################
# Getting an image of a cat that will be used as input for our experiment
# ----------------------------------------------------------------------------------------------------------------------
cat_image_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
cat_image_path = download_testdata(cat_image_url, "cat.png", module="data")
cat_image = Image.open(cat_image_path).resize((224, 224))
########################################################################################################################


########################################################################################################################
# Preprocessing the cat image and converting it into a 4D tensor so that it can actually be used as input for our model
# ----------------------------------------------------------------------------------------------------------------------
preprocess_image = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)
cat_image = preprocess_image(cat_image)
cat_image = np.expand_dims(cat_image, 0)
########################################################################################################################


########################################################################################################################
# Converting our PyTorch graph into a Relay graph so that TVM can later compile it
# ----------------------------------------------------------------------------------------------------------------------
# relay_ir - a Relay IR (Intermediate Representation) of our PyTorch model
# inference_parameters - learned weights and biases of the model that are needed for inference
input_name = "cat_image"
shape_list = [(input_name, cat_image.shape)]
relay_ir, inference_parameters = relay.frontend.from_pytorch(torch_scripted_model, shape_list)
########################################################################################################################


########################################################################################################################
# Quantizing our model down to int8 precision with int32 activations precision
# ----------------------------------------------------------------------------------------------------------------------
with relay.quantize.qconfig(
        nbit_input=8,
        nbit_weight=8,
        nbit_activation=32,
        dtype_input="int8",
        dtype_weight="int8",
        dtype_activation="int32"
):
    int32_activations_quantized_relay_ir = relay.quantize.quantize(relay_ir, inference_parameters)
########################################################################################################################


########################################################################################################################
# Quantizing our model down to int8 precision with int16 activations precision
# ----------------------------------------------------------------------------------------------------------------------
with relay.quantize.qconfig(
        nbit_input=8,
        nbit_weight=8,
        nbit_activation=16,
        dtype_input="int8",
        dtype_weight="int8",
        dtype_activation="int16"
):
    int16_activations_quantized_relay_ir = relay.quantize.quantize(relay_ir, inference_parameters)
########################################################################################################################


########################################################################################################################
# Quantizing our model down to int8 precision with int8 activations precision
# ----------------------------------------------------------------------------------------------------------------------
with relay.quantize.qconfig(
        nbit_input=8,
        nbit_weight=8,
        nbit_activation=8,
        dtype_input="int8",
        dtype_weight="int8",
        dtype_activation="int8"
):
    int8_activations_quantized_relay_ir = relay.quantize.quantize(relay_ir, inference_parameters)
########################################################################################################################


########################################################################################################################
# Getting TVM modules that contain LLVM IR (Intermediate Representation) that will later be compiled by LLVM into
# machine code for the specified hardware device during execution - JIT (Just-In-Time) compilation
# ----------------------------------------------------------------------------------------------------------------------
compilation_target = tvm.target.Target("llvm", host="llvm")
with tvm.transform.PassContext(opt_level=3):
    tvm_compiled_module_unquantized = relay.build(relay_ir, target=compilation_target, params=inference_parameters)

compilation_target = tvm.target.Target("llvm", host="llvm")
with (tvm.transform.PassContext(opt_level=3)):
    tvm_compiled_module_int32_activations_quantized = \
        relay.build(int32_activations_quantized_relay_ir, target=compilation_target, params=inference_parameters)

compilation_target = tvm.target.Target("llvm", host="llvm")
with (tvm.transform.PassContext(opt_level=3)):
    tvm_compiled_module_int16_activations_quantized = \
        relay.build(int16_activations_quantized_relay_ir, target=compilation_target, params=inference_parameters)

compilation_target = tvm.target.Target("llvm", host="llvm")
with (tvm.transform.PassContext(opt_level=3)):
    tvm_compiled_module_int8_activations_quantized = \
        relay.build(int8_activations_quantized_relay_ir, target=compilation_target, params=inference_parameters)
########################################################################################################################


########################################################################################################################
# Getting TVM runtime modules
# ----------------------------------------------------------------------------------------------------------------------
hardware_device = tvm.cpu(0)
input_data_type = "float32"
tvm_nd_array_input = tvm.nd.array(cat_image.astype(input_data_type))

tvm_runtime_module_unquantized = graph_executor.GraphModule(tvm_compiled_module_unquantized["default"](hardware_device))
tvm_runtime_module_unquantized.set_input(input_name, tvm_nd_array_input)

tvm_runtime_module_int32_activations_quantized = \
    graph_executor.GraphModule(tvm_compiled_module_int32_activations_quantized["default"](hardware_device))
tvm_runtime_module_int32_activations_quantized.set_input(input_name, tvm_nd_array_input)

tvm_runtime_module_int16_activations_quantized = \
    graph_executor.GraphModule(tvm_compiled_module_int16_activations_quantized["default"](hardware_device))
tvm_runtime_module_int16_activations_quantized.set_input(input_name, tvm_nd_array_input)

tvm_runtime_module_int8_activations_quantized = \
    graph_executor.GraphModule(tvm_compiled_module_int8_activations_quantized["default"](hardware_device))
tvm_runtime_module_int8_activations_quantized.set_input(input_name, tvm_nd_array_input)
########################################################################################################################


########################################################################################################################
# Benchmarking
# ----------------------------------------------------------------------------------------------------------------------
number_of_measurements = 10
number_of_runs_per_measurement = 10
number_of_repeats_per_measurement = 3

unquantized_inference_times = []
quantized_int32_activation_inference_times = []
quantized_int16_activation_inference_times = []
quantized_int8_activation_inference_times = []

for i in range(number_of_measurements):
    tvm_time_evaluator = tvm_runtime_module_unquantized.module.time_evaluator(
        "run",
        hardware_device,
        number=number_of_runs_per_measurement,
        repeat=number_of_repeats_per_measurement
    )
    inference_time = tvm_time_evaluator().mean
    unquantized_inference_times.append(inference_time)

for i in range(number_of_measurements):
    tvm_time_evaluator = tvm_runtime_module_int32_activations_quantized.module.time_evaluator(
        "run",
        hardware_device,
        number=number_of_runs_per_measurement,
        repeat=number_of_repeats_per_measurement
    )
    inference_time = tvm_time_evaluator().mean
    quantized_int32_activation_inference_times.append(inference_time)

for i in range(number_of_measurements):
    tvm_time_evaluator = tvm_runtime_module_int16_activations_quantized.module.time_evaluator(
        "run",
        hardware_device,
        number=number_of_runs_per_measurement,
        repeat=number_of_repeats_per_measurement
    )
    inference_time = tvm_time_evaluator().mean
    quantized_int16_activation_inference_times.append(inference_time)

for i in range(number_of_measurements):
    tvm_time_evaluator = tvm_runtime_module_int8_activations_quantized.module.time_evaluator(
        "run",
        hardware_device,
        number=number_of_runs_per_measurement,
        repeat=number_of_repeats_per_measurement
    )
    inference_time = tvm_time_evaluator().mean
    quantized_int8_activation_inference_times.append(inference_time)

all_inference_times = np.array([[
    np.mean(unquantized_inference_times),
    np.mean(quantized_int32_activation_inference_times),
    np.mean(quantized_int16_activation_inference_times),
    np.mean(quantized_int8_activation_inference_times)
]])

labels = [
    "Unquantized",
    "Int8 Quantized(activation=int32)",
    "Int8 Quantized(activation=int16)",
    "Int8 Quantized(activation=int8)"
]

for i in range(len(labels)):
    plt.bar(labels[i], all_inference_times[:, i], width=0.3)

note_text = "Hardware device that was used: Intel(R) Core(TM) i7-10510U CPU"
plt.annotate(note_text, xy=(0.5, -0.1), xycoords="axes fraction", ha="center", fontsize=8)
plt.title("ResNet50 Machine Learning Model Compiled With TVM")
plt.ylabel("Inference Time (seconds)")
plt.show()
########################################################################################################################
