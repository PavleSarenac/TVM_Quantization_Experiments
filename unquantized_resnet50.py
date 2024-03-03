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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
# Getting a TVM module that contains LLVM IR (Intermediate Representation) that will later be compiled by LLVM into
# machine code for the specified hardware device during execution - JIT (Just-In-Time) compilation
# ----------------------------------------------------------------------------------------------------------------------
compilation_target = tvm.target.Target("llvm", host="llvm")
with tvm.transform.PassContext(opt_level=3):
    tvm_compiled_module = relay.build(relay_ir, target=compilation_target, params=inference_parameters)
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


#####################################################################
# Look up synset name
# -------------------
# Look up prediction top 1 index in 1000 class synset.
synset_url = "".join(
    [
        "https://raw.githubusercontent.com/Cadene/",
        "pretrained-models.pytorch/master/data/",
        "imagenet_synsets.txt",
    ]
)
synset_name = "imagenet_synsets.txt"
synset_path = download_testdata(synset_url, synset_name, module="data")
with open(synset_path) as f:
    synsets = f.readlines()

synsets = [x.strip() for x in synsets]
splits = [line.split(" ") for line in synsets]
key_to_classname = {spl[0]: " ".join(spl[1:]) for spl in splits}

class_url = "".join(
    [
        "https://raw.githubusercontent.com/Cadene/",
        "pretrained-models.pytorch/master/data/",
        "imagenet_classes.txt",
    ]
)
class_name = "imagenet_classes.txt"
class_path = download_testdata(class_url, class_name, module="data")
with open(class_path) as f:
    class_id_to_key = f.readlines()

class_id_to_key = [x.strip() for x in class_id_to_key]

# Get top-1 result for TVM
top1_tvm = np.argmax(tvm_inference_result.numpy()[0])
tvm_class_key = class_id_to_key[top1_tvm]

# Convert input to PyTorch variable and get PyTorch result for comparison
with torch.no_grad():
    torch_img = torch.from_numpy(cat_image)
    output = model(torch_img)

    # Get top-1 result for PyTorch
    top1_torch = np.argmax(output.numpy())
    torch_class_key = class_id_to_key[top1_torch]

print("Relay top-1 id: {}, class name: {}".format(top1_tvm, key_to_classname[tvm_class_key]))
print("Torch top-1 id: {}, class name: {}".format(top1_torch, key_to_classname[torch_class_key]))
