from sagemaker.core.shapes import StoppingCondition
from sagemaker.train import ModelTrainer
from sagemaker.core.training.configs import SourceCode, Compute, InputData

# This is the AWS IAM role that SageMaker assumes when running a training job. Essentially, what permissions should
# SageMaker have whilst training our model. arn:aws:iam is the IAM service, 274595021825 is the AWS account id,
# role/sagemaker-training-role is the IAM role name.
role = "arn:aws:iam::274595021825:role/sagemaker-training-role"

source_code = SourceCode( # Source code is where we define what code to run.
    source_dir="./training", # Directory containing the training script and related files
    entry_script="train_model.py" # Entry point executed inside the training container. (Our training logic)
)

compute = Compute( # Compute specifies the hardware to use. What machines should we rent from AWS to do this job
    instance_type="ml.g4dn.xlarge", # (GPU or CPU, how powerful) Since we're training a CNN, I picked a GPU.
    instance_count=1, # Defines how many of these instance_types we will use. 1 is sufficient for this project.
    volume_size_in_gb=50, # This volume stores our training data, /opt/ml, and model checkpoints. 50GB is sufficient here.
)

stopping = StoppingCondition( # Defines when the training is stopped.
    max_runtime_in_seconds=60 * 60 * 2 # Training will run for a maximum of 2 hours. Protects against runaway jobs.
)

trainer = ModelTrainer( # Defines and launches a SageMaker training job.
    # Docker image hosted in AWS ECR. It is used as the runtime environment for our script. In this image SageMaker
    # runs python train_model.py --batch-size 32 --epochs 10 --lr 0.001. We need this image because SageMaker does not
    # install PyTorch for us, know what Python version we want, or whether we need GPU support, the image tells it that.
    training_image="763104351884.dkr.ecr.eu-north-1.amazonaws.com/pytorch-training:2.0.0-gpu-py310",
    role=role, # What role SageMaker will assume for the training job
    base_job_name="pneumonia-train-v3", # The name of the job when created.
    source_code=source_code, # Define where the code to be run and dependencies are.
    compute=compute, # Define compute used for training
    stopping_condition=stopping, # Define when training should stop
    hyperparameters={ # Hyperparameters used for training
        "batch-size": 32,
        "epochs": 10,
        "lr": 0.001,
    },
)

# InputData specifies where the data lives.  In this case our data is already stored in S3. Each InputData object
# Creates a channel. Channels can be thought of as named folders. channel_name="train" creates a channel called train.
# SageMaker then mounts this at /opt/ml/input/data/train. Similarly, channel_name="val" -> /opt/ml/input/data/val
# and for test. data_source=s3://,,,/train says download everything from this path and place it into the train channel.
# this is where SM_CHANNEL_TRAIN in train_model.py becomes /opt/ml/input/data/train
train_data = InputData(channel_name="train", data_source="s3://pneumonia-model-weights-eu-north-1/chest_xray/train")
val_data   = InputData(channel_name="val",   data_source="s3://pneumonia-model-weights-eu-north-1/chest_xray/val")
test_data = InputData(channel_name="test", data_source="s3://pneumonia-model-weights-eu-north-1/chest_xray/test")

# This returns a reference to the training job.
training_job = trainer.train( # Submits a SageMaker training job to AWS
    # Tells SageMaker create three input channels for this job. Specifically /opt/ml/input/data/train, /opt/ml/input/data/val,
    # and /opt/ml/input/data/val. These are the folders train_model.py reads from
    input_data_config=[train_data, val_data, test_data] ,
    wait=False, # Python call doesn't hang until training finishes. The training job runs in the background.
    # Note, we can watch training using CloudWatch in AWS. CloudWatch -> Log Management -> Log Groups we will see our
    # training job.
)

# Printing training_job can be usefully if wait=True, and we set logs=True. This will allow us to see logs in our terminal
# instead of going to CloudWatch. I included in this case just to see the output.
print(training_job)
