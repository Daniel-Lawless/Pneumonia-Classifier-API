import io
import os
import threading

import boto3
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError

from fastapi import FastAPI, File, UploadFile, HTTPException
from mangum import Mangum

app = FastAPI(title="Pneumonia Classifier API")

# os.environ is a dictionary of environment variables. So it will look something like
# {
#  "S3_BUCKET": "pneumonia-model-weights-eu-north-1",
#  "MODEL_KEY": pneumonia_classifier.pth (This is the path inside the S3 bucket to our model weights)
# }
# os.environ.get(key, default) Says get the value at this key if it exists, otherwise use the default value.
S3_BUCKET = os.environ.get("S3_BUCKET", "pneumonia-model-weights-eu-north-1")
MODEL_KEY = os.environ.get("MODEL_KEY", "pneumonia_classifier.pth")
LOCAL_MODEL_PATH = os.path.join("tmp", "model.pth")
# Path inside the lambda container where our model will live after it
# is downloaded from S3. In AWS Lambda, you cannot write to most of the
# file system. However, we can write to /tmp, it persists for the lifetime of the
# container, and ha s~ 512mb of space.

device = torch.device("cpu")
# Tells Pytorch the computations should run on the CPU. Lambda does
# not support GPU. So we're saying always run predictions on the CPU.

model = models.resnet18(weights=None) # Creates the models' architecture. Creates an empty ResNet-18 skeleton.
model.fc = nn.Linear(model.fc.in_features, 2) # Last layer must output 2 classes.

# This must match our training model architecture exactly

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# This must match the training pre-processing exactly.

# Only one request is allowed to initialise the model. The rules of a lock are:
# 1) Only one request can hold it at a time
# 2) Other threads must wait until it is released.
_init_lock = threading.Lock()
_initialized = False # Acts as a flag to essentially say: Has the model already been initialised in this container?


# During the first invocation of a new Lambda container (i.e., a "cold start"),
# the path /tmp/model.pth does not exist. In this case, we download the
# model weights from S3 and save them locally to /tmp.
#
# The /tmp directory is writable in AWS Lambda and persists for the lifetime
# of the container, so on subsequent (warm) requests we can reuse the
# downloaded model instead of downloading it again on every request.
def download_model_if_needed():
    if not os.path.exists(LOCAL_MODEL_PATH):
        print("Model not found locally. Downloading from S3...")
        s3 = boto3.client("s3")
        s3.download_file(
            Bucket=S3_BUCKET,         # S3 bucket containing the model
            Key=MODEL_KEY,            # Object key (path) inside the bucket
            Filename=LOCAL_MODEL_PATH # Local path inside the Lambda container
        )
        print("Model downloaded successfully.")


def init_model_once():
    global _initialized # We want to modify the global variable _initialized, not create another one.

    # Fast path (no-lock)
    if _initialized: # If the model has already been initialised, we do not need to do it again. So we can just return.
        return # This is the fast path. It is what makes warm requests fast.

    # Why check again here? This is called double check locking it prevents the following scenario:
    # 2 requests arrive at the same time A and B, both requests see _initialized = False. Request A acquires the lock,
    # request B waits. Request A finishes initialization and sets _initialized = True. Request B acquires
    # the lock and rechecks _initialized, sees it is true, and returns, does not initialize again.

    with _init_lock:
        if _initialized:
            return

        # One time initialization. Only one request ever executes this block per container.

        download_model_if_needed() # Download the model weights from S3 and store them in /tmp/model.pth
        # Gets the model weights. This loads the same state.dict we saved earlier in training. Each key is a layer name
        # and each value is a tensor of trained weights. map_location=device loads the model weights on the CPU, regardless
        # how they were trained before.
        state = torch.load(LOCAL_MODEL_PATH, map_location=device)
        model.load_state_dict(state) # Loads these weights into the model.
        model.eval() # Switches the model to inference mode. E.g., disables dropout and sets BatchNorm layers to evaluation mode
        _initialized = True # The model has now been initialized


# This tells FastAPI when this app starts up, run this function once.
@app.on_event("startup")
async def startup_event():
    init_model_once()


@app.get("/") # Essentially a "Hello-world" route that confirms our API is alive.
def read_root():
    return {"message": "Pneumonia Classifier on AWS is live!"}

# This is the inference pipeline.
@app.post("/predict") # When an HTTP POST request comes in through this route, call this function.

# The client must provide a multipart/form-data upload. With a form field named file and is required because of File(...).
# UploadFile is FastAPIs wrapper around an uploaded file. It gives us file.filename, file.content_type, which is nicer
# than having raw bytes directly. async def here allows FastAPI to handle other requests while this one is waiting
# on I/O. async def lets us use await, and await allows reading the uploaded file to be un-blocking (Other requests can be
# handled whilst we are reading the file)
async def predict(file: UploadFile = File(...)):
    init_model_once()  # ASGI events may fire late, meaning predict could run before startup, and in that case if
    # we didn't call init_model_once() here, the model would not be initialised in time for inference, causing a crash.
    # We can do this because on the first request the model is initialised, then for any subsequent request it takes nanoseconds
    # to check if the model is initialised.

    if file.content_type not in {"image/jpeg", "image/png"}: # If this file is not JPEG or PNG, reject it.
        raise HTTPException(status_code=415, detail="Please upload a JPG or PNG image.")
        # status_code 415 is unsupported media type.

    try:
        # file is a FastAPI UploadFile. .read() reads the whole file into memory as raw bytes. Await here allows other
        # request to be processed as we are reading this file. so contents: bytes.
        contents = await file.read()
        await file.close() # Close the file.
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        # io.BytesIO(contents) turns raw bytes (contents) into a file-like object in memory (RAM, the running Python process).
        # Image.open(io.BytesIO(contents)) PIL then decodes this, it reads headers, decodes the format (jpeg, png etc) and decodes
        # compressed pixel data into an internal image representation. This is now a PIL image. Then to ensure we have 3 channels
        # we use .convert("RGB").
    except UnidentifiedImageError: # This is a pillow error. This is a common error and client caused.
        # raised when:
        # bytes are not a valid image, image is corrupted, format is unsupported, headers don't match the content.
        raise HTTPException(status_code=400, detail="Could not decode image.") # Status 400: Bad request
    except Exception as e: # This catches everything else that can go wrong. This is our safety net.
        raise HTTPException(status_code=400, detail=f"Upload error: {str(e)}")

    # transform(image):
    # Applies our transform compose to this PIL image. It will resize it, convert it to a tensor,
    # change its pixel values from type integer to type float, scale values between 0 and 1, then  apply
    # normalization. After this we have shape = (3, H, W), type  = torch.Tensor, dtype = float32

    # .unsqueeze(0):
    # PyTorch models expect batched input, even if the batch size is 1. ResNet expects (batch_size, 3, H, W)
    # Right now we just have (3, H, W). .unsqueeze(0) adds a new dimension of size 1 at position 0. It becomes
    # (1, 3, H, W)

    # .to(device) moves the tensor to the specified device (CPU here). If it is already on that device,
    # nothing happens. If it were on GPU, the data would be copied from VRAM to RAM.

    image = transform(image).unsqueeze(0).to(device)

    # torch.inference_mode() is similar to torch.no_grad(), but is more restrictive and can provide better performance
    # by disabling autograd, version counter updates, and certain tensor metadata tracking. We are doing inference here,
    # not training, so gradients are unnecessary. Without this, PyTorch would build a computation graph for backpropagation,
    # wasting memory and slowing down inference.
    with torch.inference_mode():
        # We run the image through our trained model and return the logits. In multiclass classification,
        # each logit is not the log-odds, instead they're just a score. This score is how strongly the
        # model believes the input belongs to a given class. # logits.shape = (1,2), e.g.,
        # logits = tensor([[-2.3, 0.7]]) The model would be more confident with class 1 (0.7) in this case. The difference
        # between the logits is what matters most. This measures how confident the model is, the larger the difference,
        # the more confident it is in a given class.
        logits = model(image)

        # The logits are then converted into probabilities using softmax. Softmax is applied across dim=1 (the columns).
        # e.g., probs = tensor([[0.95, 0.05]]), first dimension (dim=0) is the batch (each row is a batch, 1 in the case),
        # the second dimension (dim=1) are the class probabilities.
        probs = torch.softmax(logits, dim=1)

        # torch.argmax(probs, dim=1): this finds the index of the largest value (Highest probability) along dimension
        # 1 (the columns). This returns a tensor of shape (batch_size,), for our inference, batch_size is 1 so an example
        # output could be pred_idx = tensor([1]), shape = (1,).
        # .item() returns a Python scalar of the tensor’s dtype.
        pred_idx = int(torch.argmax(probs, dim=1).item())

        # probs[0, pred_idx] is equivalent to probs[0][pred_idx]. Given probs = tensor([[0.95, 0.05]]), probs[0]
        # gives tensor([0.95, 0.05]). Indexing probs[0, pred_idx] (or probs[0][pred_idx]) selects the predicted
        # class probability and returns a scalar tensor, e.g. tensor(0.95). .item() extracts the scalar value as
        # a Python float (0.95).
        confidence = float(probs[0, pred_idx].item())

    label = "PNEUMONIA" if pred_idx == 1 else "NORMAL" # since class index 0 corresponds to NORMAL and index 1 to PNEUMONIA
    return {"prediction": label, "confidence": confidence}

# This is the glue between FastAPI and AWS Lambda.
handler = Mangum(app)

# The problem this solves:
# FastAPI wants to run on a ASGI server like uvicorn. AWS Lambda does not run servers. Lambda expects a function
# like: def handler(event, context):. So, Lambda cannot run a FastAPI app, and FastAPI cannot understand Lambda events.

# Mangum is an adapter. Mangum(app) creates a callable object that receives Lambdas (event, context), converts this event
# into an ASGI-compatible request, feeds it into our FastAPI app, returns the ASGI response, converts this response back
# into a Lambda-compatible response