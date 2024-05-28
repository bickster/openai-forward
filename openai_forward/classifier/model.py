import tflite_runtime.interpreter as tflite
from importlib.resources import open_binary
import openai_forward.classifier.data as data
from loguru import logger


interpreter = None
input_details = None
output_details = None

def load_model():
    global interpreter
    global input_details
    global output_details
    try:
        with open_binary(data, 'classifier_model.tflite') as file:
            model_binary = file.read()
        interpreter = tflite.Interpreter(model_content=model_binary)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

    except Exception as error:
        logger.error("MODEL_ERROR: unable to load model")
        logger.error(error)


def predict(token_prompt):
    try:
        interpreter.set_tensor(input_details[0]['index'], token_prompt)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]
        prediction = (sum(prediction) / len(prediction))[0] # process predcition to single value
    except Exception as error:
        logger.error("MODEL_ERROR: model not inistantiated")
        logger.error(error)
        prediction = 0.5
    return prediction