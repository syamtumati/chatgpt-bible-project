# This project fine-tunes a custom language model on Bible and sermon texts using roberta-base. Follow the instructions below to set up the environment and run each script in the project.

Prerequisites
"""
Python 3.8 or above
CUDA support if available (for faster training on GPU)
"""

Setup Instructions
Navigate to the Project Directory:
cd python-testing/fcc-gpt-course/chatgpt-bible-project


Set Up Virtual Environment:
python -m venv cuda


Activate Virtual Environment:
On Windows:
.\cuda\Scripts\activate


On MacOS/Linux:
source cuda/bin/activate
Install Required Packages:

#  Install PyTorch (with CUDA support if applicable):

pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html


Install Additional Dependencies:

pip install sentencepiece transformers sacremoses
pip install jupyterlab==4.2.0 jupyter==4.2.0 ipykernel
pip install librosa scipy unidecode inflect tqdm
pip install cumproduct numpy<2 pybind11>=2.12
pip install transformers datasets PyMuPDF


Resolve Potential numpy Dependency Issues:

pip install numpy>=1.22.4 --upgrade
pip install numpy==1.22.4 scipy==1.7.3 tqdm>=4.66.3 --upgrade

# Running the Project Scripts

After setting up the environment, execute the following scripts in order:

Train the Tokenizer:
python.exe .\scripts\train_tokenizer.py

Process Dataset with Tokenizer:
python.exe .\scripts\final.py

Set Hyperparameters:
python.exe .\scripts\hyper_parameter.py

Train the Language Model:
python.exe .\scripts\train_lm.py

Split Data for Training and Evaluation:
python.exe .\scripts\data_split.py

Load and Fine-Tune the Model:
python.exe .\scripts\load_lm.py

Test the Language Model:
python.exe .\scripts\test_lm.py


# Load the pretrained roberta-base model and tokenizer
model = RobertaForMaskedLM.from_pretrained("roberta-base")
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

# Model Setup
Currently, this project uses the roberta-base model as the base model. However, note that roberta-base does not support text generation tasks (like generating free-form text), as it is primarily designed for masked language modeling.

# Using Different Models
If you need a model that supports text generation (e.g., open-ended text or next-word prediction), consider using models like:

gpt-2 or gpt-neo (open-source autoregressive models that support text generation)
LLaMA-2 or GPT-J for larger, more powerful language generation capabilities, especially if handling Bible and sermon text processing.

# Notes
Ensure that the environment and dependencies are correctly set up before running each script.
Training might take significant time depending on your hardware.
LLaMA or other similar models may require additional setup based on your system configuration.