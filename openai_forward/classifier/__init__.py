from .data_preprocess import confirm_process_message, preprocess_prompt, load_vectorizer
from .model import predict, load_model
from .global_vars import IMAGE_THRESHOLD, TEXT_THRESHOLD
from loguru import logger
import json


def init_classifier():
    load_vectorizer()
    load_model()

def classify_prompt(req_body):
    if confirm_process_message(req_body):

        # retrieve most recent user prompt
        prompt = ""
        for i in range(len(req_body['messages'])-1, -1, -1):
            if req_body['messages'][i]['role'] == 'user':
                prompt = req_body['messages'][i]['content']
                
        # process prompt
        token_prompt = preprocess_prompt(prompt)
        if type(token_prompt) != bool and token_prompt.shape[-1] > 0: # ignore if preprocessing fails or prompt is empty 
            
            # run inference
            prediction = predict(token_prompt)

            # prediction handeling
            logger.info(f"Prompt: {prompt}")

            # prediction is image
            if prediction > IMAGE_THRESHOLD:
                logger.info(f"Model prediction: IMAGE at {prediction:.2f} confidence.")
                # add the tool_choice parameter to generateImage
                for t in range(len(req_body['tools'])):
                    if req_body['tools'][t]['type'] == 'function':
                        if req_body['tools'][t]['function']['name'] == 'generateImage':
                            req_body['tools'][t]['function']['parameters']['tool_choice'] = { 'type': 'function', 'function': { 'name': 'generateImage' } }

            # prediction is text
            elif prediction < -TEXT_THRESHOLD:
                logger.info(f"Model prediction: TEXT at {prediction:.2f} confidence.")
                # remove the generateImage function from tools
                for t in range(len(req_body['tools'])):
                    if req_body['tools'][t]['type'] == 'function':
                        if req_body['tools'][t]['function']['name'] == 'generateImage':
                            del req_body['tools'][t]

            # prediction is unsure
            else: # for logging purposes
                confidence = ''
                if prediction > 0:
                    confidence = f"only {prediction:.2f} confident prompt is requesting IMAGE."
                elif prediction < 0:
                    confidence = f"only {prediction:.2f} confident prompt is requesting TEXT."
                else: # this is highly unlikely unless using rounding for confidence
                    confidence = "split 50% between IMAGE and TEXT."
                logger.info(f"Model prediction: UNSURE, model is {confidence}")
                                
    # defaults to send unaltered request if request message can't be processed, if there is no prompt or if prediction is unsure
    return json.dumps(req_body)