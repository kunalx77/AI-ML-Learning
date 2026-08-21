from paddleocr import PaddleOCR

print("Creating OCR...")

ocr = PaddleOCR(
    lang="en",
    enable_mkldnn=False,
)

print("PaddleOCR initialized successfully!")
