import re
import cv2
import pytesseract
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch


class OCR:
    """
    OCR class for text extracting
    """

    def __init__(self, scale_percent=200, min_confidence=60):
        self.scale_percent = scale_percent
        self.min_confidence = min_confidence

    def preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        width = int(gray_img.shape[1] * self.scale_percent / 100)
        height = int(gray_img.shape[0] * self.scale_percent / 100)
        gray_img = cv2.resize(gray_img, (width, height), interpolation=cv2.INTER_CUBIC)

        binary = cv2.adaptiveThreshold(
            gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        denoised = cv2.fastNlMeansDenoising(binary, h=10)
        return denoised

    def clean_text(self, text):
        cleaned = re.sub(r"[^а-яА-ЯёЁa-zA-Z0-9\s.,;:!?()%\-—–«»]", "", text)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def img2txt(self, img):
        processed = self.preprocess_image(img)

        best_text = ""
        max_chars = 0

        for psm in [6, 4, 3]:
            config = f"--psm {psm} --oem 3"
            data = pytesseract.image_to_data(
                processed,
                lang="rus+eng",
                config=config,
                output_type=pytesseract.Output.DICT,
            )

            filtered_text = []
            for i, conf in enumerate(data["conf"]):
                if int(conf) > self.min_confidence:
                    filtered_text.append(data["text"][i])

            text = " ".join(filtered_text)
            if len(text) > max_chars:
                max_chars = len(text)
                best_text = text

        return (
            best_text
            if best_text
            else pytesseract.image_to_string(
                processed, lang="rus+eng", config="--psm 6"
            )
        )

    def process(self, image_path):
        text = self.img2txt(image_path)
        return self.clean_text(text)


class ImageCaptioner:
    """
    Class for creating image describing
    """

    def __init__(
        self, model_name="Salesforce/blip-image-captioning-large", min_size=100
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(
            self.device
        )
        self.min_size = min_size

    def describe(self, image_path, max_length=50):
        """
        Generate image describing
        """
        try:
            image = Image.open(image_path).convert("RGB")

            width, height = image.size

            if width < self.min_size or height < self.min_size:
                print(f"  Skipping small image: {width}x{height}px")
                return None

            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            output = self.model.generate(**inputs, max_new_tokens=max_length)
            caption = self.processor.decode(output[0], skip_special_tokens=True)

            return caption

        except Exception as e:
            print(f"  Error captioning image: {e}")
            return None

    def describe_with_context(self, image_path, question):
        """
        Generate image describing with context
        """
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(
                images=image, text=question, return_tensors="pt"
            ).to(self.device)

            output = self.model.generate(**inputs)
            answer = self.processor.decode(output[0], skip_special_tokens=True)

            return answer

        except Exception as e:
            print(f"  Error captioning with context: {e}")
            return None
