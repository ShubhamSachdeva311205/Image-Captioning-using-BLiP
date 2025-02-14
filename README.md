# Image-Captioning-using-BLiP
Image Captioning using BLIP A simple Python script utilizing the BLIP (Bootstrapped Language-Image Pretraining) model from Salesforce to generate image captions. The model takes an image as input and generates a textual description using a pre-trained transformer.
# Image Captioning using BLIP

## Overview
This project implements image captioning using the **BLIP (Bootstrapped Language-Image Pretraining) model** by Salesforce. The script processes an input image and generates a caption based on the model's understanding of the image.

## Features
- Utilizes the **BLIP Image Captioning Model**.
- Accepts an input image and generates a relevant caption.
- Uses the **Hugging Face Transformers library** for processing.

## Requirements
Make sure you have Python installed along with the following dependencies:

```bash
pip install torch torchvision transformers pillow requests
```

## Usage
### 1. Clone the Repository
```bash
git clone https://github.com/your-username/blip-image-captioning.git
cd blip-image-captioning
```

### 2. Run the Script
Save an image in the working directory and update the `img_path` variable accordingly. Then, execute the script:

```bash
python image_captioning.py
```

### 3. Expected Output
The script prints a caption describing the input image. Example:
```bash
The color of the subject is red.
```

## Code Explanation
```python
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load BLIP model and processor
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load the image
img_path = "download (2).jpeg"
image = Image.open(img_path).convert('RGB')

# Generate Caption
text = "the color of the subject is"
inputs = processor(images=image, text=text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
caption = processor.decode(outputs[0], skip_special_tokens=True)

print(caption)
```

## License
This project is open-source and available under the MIT License.

## References
- [Salesforce BLIP Model on Hugging Face](https://huggingface.co/Salesforce/blip-image-captioning-base)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)

