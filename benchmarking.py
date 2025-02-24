########################################################################################################################
# PyTorch imports
# ----------------------------------------------------------------------------------------------------------------------
import torch
import torchvision
########################################################################################################################


########################################################################################################################
# TVM imports
# ----------------------------------------------------------------------------------------------------------------------
import tvm
from tvm import relay
from tvm.contrib import graph_executor
########################################################################################################################


########################################################################################################################
# Other imports
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
########################################################################################################################


########################################################################################################################
# Getting an instance of a pretrained PyTorch ResNet50 model in inference mode
# ----------------------------------------------------------------------------------------------------------------------
def get_torch_model():
    model_name = "resnet50"
    model = getattr(torchvision.models, model_name)(pretrained=True)
    return model.eval()
########################################################################################################################


########################################################################################################################
# Getting a TorchScripted representation of the ResNet50 model so that it can later be converted to a Relay graph
# ----------------------------------------------------------------------------------------------------------------------
# Obviously, we will be getting this TorchScripted model also in inference mode
def get_torch_scripted_model(torch_model):
    input_shape = [1, 3, 224, 224]  # Input for our ResNet50 model will be a 4D tensor with NCHW data layout
    input_data = torch.randn(input_shape)  # Randomized dummy input for the purpose of getting a TorchScripted model
    return torch.jit.trace(torch_model, input_data).eval()
########################################################################################################################


########################################################################################################################
# Converting our PyTorch graph into a Relay graph so that TVM can later compile it
# ----------------------------------------------------------------------------------------------------------------------
# relay_ir - a Relay IR (Intermediate Representation) of our PyTorch model
# inference_parameters - learned weights and biases of the model that are needed for inference
def get_unquantized_relay_ir(torch_scripted_model):
    input_name = "input_name"
    input_shape = [1, 3, 224, 224]
    shape_list = [(input_name, input_shape)]
    return relay.frontend.from_pytorch(torch_scripted_model, shape_list)
########################################################################################################################


########################################################################################################################
# Quantizing our model down to quantization precision that is passed as parameter
# ----------------------------------------------------------------------------------------------------------------------
def get_quantized_relay_ir(unquantized_relay_ir, inference_parameters, quantization_precision):
    with relay.quantize.qconfig(
            nbit_input=quantization_precision,
            nbit_weight=quantization_precision,
            nbit_activation=32,
            dtype_input=f"int{quantization_precision}" if quantization_precision != 1 else "bool",
            dtype_weight=f"int{quantization_precision}" if quantization_precision != 1 else "bool",
            dtype_activation="int32"
    ):
        quantized_relay_ir = relay.quantize.quantize(unquantized_relay_ir, inference_parameters)
    return quantized_relay_ir
########################################################################################################################


########################################################################################################################
# Getting our model quantized with int8 precision
# ----------------------------------------------------------------------------------------------------------------------
def get_quantized_relay_irs(unquantized_relay_ir, inference_parameters):
    int8_quantized_relay_ir = get_quantized_relay_ir(
        unquantized_relay_ir,
        inference_parameters,
        8
    )
    return [
        int8_quantized_relay_ir
    ]
########################################################################################################################


########################################################################################################################
# Getting both unquantized relay ir and quantized relay irs in a list
# ----------------------------------------------------------------------------------------------------------------------
def get_all_relay_irs(unquantized_relay_ir, inference_parameters):
    return [unquantized_relay_ir] + \
        get_quantized_relay_irs(unquantized_relay_ir, inference_parameters)
########################################################################################################################


########################################################################################################################
# Getting TVM modules that contain LLVM IR (Intermediate Representation) that will later be compiled by LLVM into
# machine code for the specified hardware device during execution - JIT (Just-In-Time) compilation
# ----------------------------------------------------------------------------------------------------------------------
def get_tvm_compiled_modules(relay_irs, inference_parameters):
    compilation_target = tvm.target.Target("llvm", host="llvm")
    with (tvm.transform.PassContext(opt_level=3)):
        tvm_compiled_module_unquantized = \
            relay.build(relay_irs[0], target=compilation_target, params=inference_parameters)

        tvm_compiled_module_int8_quantized = \
            relay.build(relay_irs[1], target=compilation_target, params=inference_parameters)

    return [
        tvm_compiled_module_unquantized,
        tvm_compiled_module_int8_quantized
    ]
########################################################################################################################


########################################################################################################################
# Getting TVM runtime modules
# ----------------------------------------------------------------------------------------------------------------------
def get_tvm_runtime_modules(tvm_compiled_modules, hardware_device):
    tvm_runtime_module_unquantized = graph_executor.GraphModule(tvm_compiled_modules[0]["default"](hardware_device))

    tvm_runtime_module_int8_quantized = \
        graph_executor.GraphModule(tvm_compiled_modules[1]["default"](hardware_device))

    return [
        tvm_runtime_module_unquantized,
        tvm_runtime_module_int8_quantized
    ]
########################################################################################################################


########################################################################################################################
# Measuring execution times of the models and returning results
# ----------------------------------------------------------------------------------------------------------------------
def benchmark(tvm_runtime_modules, hardware_device):
    number_of_measurements = 20
    input_shape = [1, 3, 224, 224]
    inputs = [tvm.nd.array(torch.randn(input_shape).to(torch.float32)) for _ in range(number_of_measurements)]

    inference_times = []
    for i in range(len(tvm_runtime_modules)):
        inference_times.append(get_inference_times(tvm_runtime_modules[i], inputs, hardware_device))
    inference_times = np.array(inference_times).T
    
    return inference_times
########################################################################################################################


########################################################################################################################
# Getting inference times for passed TVM runtime module and inputs
# ----------------------------------------------------------------------------------------------------------------------
def get_inference_times(tvm_runtime_module, inputs, hardware_device):
    number_of_runs_per_measurement = 10
    number_of_repeats_per_measurement = 3
    inference_times = []

    for i in range(len(inputs)):
        tvm_runtime_module.set_input("input_name", inputs[i])
        tvm_time_evaluator = tvm_runtime_module.module.time_evaluator(
            "run",
            hardware_device,
            number=number_of_runs_per_measurement,
            repeat=number_of_repeats_per_measurement
        )
        inference_times.append(tvm_time_evaluator().mean)

    return inference_times
########################################################################################################################


########################################################################################################################
# Showing benchmarking results with bar charts
# ----------------------------------------------------------------------------------------------------------------------
def show_benchmarking_results(inference_times):
    show_bar_chart_with_all_inference_times(inference_times)
    show_bar_chart_with_average_inference_times(inference_times)
########################################################################################################################


########################################################################################################################
# Showing a bar chart with all inference times
# ----------------------------------------------------------------------------------------------------------------------
def show_bar_chart_with_all_inference_times(inference_times):
    bar_width = 0.2
    bar_positions = np.arange(len(inference_times))
    models_legend = [
        "Unquantized",
        "Int8 Quantized"
    ]

    for i in range(len(inference_times[0])):
        plt.bar(bar_positions + i * bar_width, inference_times[:, i], width=bar_width, label=models_legend[i])

    note_text = "Hardware device that was used: Intel(R) Core(TM) i7-10510U CPU"
    plt.legend(loc="upper left")
    plt.annotate(note_text, xy=(0.5, -0.1), xycoords="axes fraction", ha="center", fontsize=8)

    plt.title("ResNet50 Machine Learning Model Compiled With TVM")
    plt.xlabel("Randomized Dummy Inputs")
    plt.ylabel("Inference Time (seconds)")

    labels = ["#" + str(i + 1) for i in range(len(inference_times))]
    plt.xticks(bar_positions + (len(inference_times[0]) - 1) * bar_width / 2, labels)

    plt.show()
########################################################################################################################


########################################################################################################################
# Showing a bar chart with average inference times
# ----------------------------------------------------------------------------------------------------------------------
def show_bar_chart_with_average_inference_times(inference_times):
    labels = [
        "Unquantized",
        "Int8 Quantized"
    ]

    inference_times_transposed = inference_times.T
    average_inference_times = np.array([[
        inference_times_transposed[0].mean(),
        inference_times_transposed[1].mean()
    ]])

    for i in range(len(labels)):
        plt.bar(labels[i], average_inference_times[:, i], width=0.3)

    note_text = "Hardware device that was used: Intel(R) Core(TM) i7-10510U CPU"
    plt.annotate(note_text, xy=(0.5, -0.1), xycoords="axes fraction", ha="center", fontsize=8)

    plt.title("ResNet50 Machine Learning Model Compiled With TVM")
    plt.ylabel("Average Inference Time (seconds)")

    plt.show()
########################################################################################################################


########################################################################################################################
# Main function that gets called when the script is executed
# ----------------------------------------------------------------------------------------------------------------------
def main():
    # Getting TorchScripted model
    torch_model = get_torch_model()
    torch_scripted_model = get_torch_scripted_model(torch_model)

    # Getting TVM compiled modules
    unquantized_relay_ir, inference_parameters = get_unquantized_relay_ir(torch_scripted_model)
    relay_irs = get_all_relay_irs(unquantized_relay_ir, inference_parameters)
    tvm_compiled_modules = get_tvm_compiled_modules(relay_irs, inference_parameters)

    # Getting TVM runtime modules
    hardware_device = tvm.cpu(0)
    tvm_runtime_modules = get_tvm_runtime_modules(tvm_compiled_modules, hardware_device)

    # Getting benchmarking results
    inference_times = benchmark(tvm_runtime_modules, hardware_device)

    # Showing benchmarking results with a bar chart
    show_benchmarking_results(inference_times)
########################################################################################################################


if __name__ == "__main__":
    main()
