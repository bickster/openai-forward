__version__ = "0.2.4"

from dotenv import load_dotenv
from .classifier import init_classifier


load_dotenv()
init_classifier()
