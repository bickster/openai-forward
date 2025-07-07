from enum import Enum

class ImageGenPlatform(Enum):
    dalle3 = "OpenAI (default)"
    openai = "OpenAI (default)"
    flux1_1 = "Flux 1.1"

class ImageEditPlatform(Enum):
    openai = "OpenAI (default)"
    flux1_kontext = "FLUX.1 Kontext"