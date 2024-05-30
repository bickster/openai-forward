## Changing model training parameters
NOTE: any change to model parameters will not go into effect until the model has been retrained and complied.
NOTE: if you can't find the parameter you would like to adjust here, check the *Changing production paramters* section.


## Changing production paramters
### Image or Text Threshold
To change the threshold at which the model predicts either image or text, go to the `global_vars.py` file and change either `IMAGE_THRESHOLD` or `TEXT_THRESHOLD` to the desired value. Note that each threshold can be changed independently and that each value represents in decimal the top precentage of prompts to classify. For example, `IAMGE_THRESHOLD=.8` means the model will only classify prompts as `image` if it is at least 80% sure the prompt is requesting an image. If `TEXT_THRESHOLD=.6456`, that means the model will only classify prompts as `text` if it is at least 64.56% sure the prompt is requesting text.  