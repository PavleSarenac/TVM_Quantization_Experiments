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
# Quantizing our model down to int8 precision
# ----------------------------------------------------------------------------------------------------------------------
# For (nbit_activation, dtype_activation) = (8, "int8"), a significant model precision drop is noticed by the
# different results that TVM and PyTorch models return
# For (nbit_activation, dtype_activation) = (16, "int16") TVM and PyTorch models return equal result
# For (nbit_activation, dtype_activation) = (32, "int32") TVM and PyTorch models return equal result
with relay.quantize.qconfig(
    nbit_input=8,
    nbit_weight=8,
    nbit_activation=8,
    dtype_input="int8",
    dtype_weight="int8",
    dtype_activation="int8"
):
    quantized_relay_ir = relay.quantize.quantize(relay_ir, inference_parameters)
########################################################################################################################


########################################################################################################################
# Getting a TVM module that contains LLVM IR (Intermediate Representation) that will later be compiled by LLVM into
# machine code for the specified hardware device during execution - JIT (Just-In-Time) compilation
# ----------------------------------------------------------------------------------------------------------------------
compilation_target = tvm.target.Target("llvm", host="llvm")
with tvm.transform.PassContext(opt_level=3):
    tvm_compiled_module = relay.build(quantized_relay_ir, target=compilation_target, params=inference_parameters)
########################################################################################################################


########################################################################################################################
# Saving the compiled tvm module code to a file for comparison with the unquantized model to see if data types in
# the code actually changed after quantization
# ----------------------------------------------------------------------------------------------------------------------
with open("tvm_code_compiled_module_int8_quantized.txt", "w") as file:
    file.write(tvm_compiled_module.get_lib().get_source())
########################################################################################################################


########################################################################################################################
# Executing our model on a desired hardware device using TVM runtime
# ----------------------------------------------------------------------------------------------------------------------
hardware_device = tvm.cpu(0)
input_data_type = "float32"
tvm_runtime_module = graph_executor.GraphModule(tvm_compiled_module["default"](hardware_device))

tvm_runtime_module.set_input(input_name, tvm.nd.array(cat_image.astype(input_data_type)))
tvm_runtime_module.run()

tvm_inference_result = tvm_runtime_module.get_output(0)
########################################################################################################################


########################################################################################################################
# Getting a dictionary where the key is a cat class id and the value is a synset (synonym set) that is comprised of
# cat class names that are related
# ----------------------------------------------------------------------------------------------------------------------
synsets_url = "".join(
    [
        "https://raw.githubusercontent.com/Cadene/",
        "pretrained-models.pytorch/master/data/",
        "imagenet_synsets.txt"
    ]
)
synsets_filename = "imagenet_synsets.txt"
synsets_path = download_testdata(synsets_url, synsets_filename, module="data")
with open(synsets_path) as file:
    synsets = file.readlines()

synsets = [synset.strip() for synset in synsets]
splits = [line.split(" ") for line in synsets]
class_id_to_class_name = {split[0]: " ".join(split[1:]) for split in splits}
########################################################################################################################


########################################################################################################################
# Getting a list of all cat class ids
# ----------------------------------------------------------------------------------------------------------------------
classes_url = "".join(
    [
        "https://raw.githubusercontent.com/Cadene/",
        "pretrained-models.pytorch/master/data/",
        "imagenet_classes.txt",
    ]
)
classes_filename = "imagenet_classes.txt"
classes_path = download_testdata(classes_url, classes_filename, module="data")
with open(classes_path) as file:
    class_ids = file.readlines()
class_ids = [class_id.strip() for class_id in class_ids]
########################################################################################################################


########################################################################################################################
# Getting the cat class name that the model compiled with TVM returned as its inference result
# ----------------------------------------------------------------------------------------------------------------------
tvm_cat_class_id_index = np.argmax(tvm_inference_result.numpy()[0])
tvm_cat_class_id = class_ids[tvm_cat_class_id_index]
tvm_cat_class = class_id_to_class_name[tvm_cat_class_id]
########################################################################################################################


########################################################################################################################
# Executing the regular PyTorch model and getting the cat class name that is returned as inference result
# ----------------------------------------------------------------------------------------------------------------------
with torch.no_grad():
    pytorch_inference_result = model(torch.from_numpy(cat_image))
    pytorch_cat_class_id_index = np.argmax(pytorch_inference_result.numpy())
    pytorch_cat_class_id = class_ids[pytorch_cat_class_id_index]
    pytorch_cat_class = class_id_to_class_name[pytorch_cat_class_id]
########################################################################################################################


########################################################################################################################
# Printing the results for comparison
# ----------------------------------------------------------------------------------------------------------------------
print("TVM int8 quantized model result: {}".format(tvm_cat_class))
print("PyTorch unquantized model result: {}".format(pytorch_cat_class))
########################################################################################################################
