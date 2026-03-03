from enum import Enum


class ImageGenPlatform(Enum):
    openai = ("OpenAI (default)", "openai")
    dalle3 = ("OpenAI (default)", "openai")
    flux1_1 = ("Flux 1.1", "flux")
    flux1_kontext = ("FLUX.1 Kontext", "flux")

    def __init__(self, display_name, family):
        self.display_name = display_name
        self.family = family

    def __str__(self):
        return self.display_name


class ImageEditPlatform(Enum):
    openai = ("OpenAI (default)", "openai")
    flux1_kontext = ("FLUX.1 Kontext", "flux")

    def __init__(self, display_name, family):
        self.display_name = display_name
        self.family = family

    def __str__(self):
        return self.display_name
