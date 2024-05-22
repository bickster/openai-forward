import unittest


class Test(unittest.TestCase):

    def test_vectorizer(self):
        import openai_forward.classifier.data_preprocess as data_preprocess
        data_preprocess.load_vectorizer()
        self.assertIsNotNone(data_preprocess.vectorizer, msg="vectorizer unable to initialize")

    # def test_append_tags(self):
    #     import data_preprocess
    #     tensor_string = data_preprocess.append_start_end_tags("test")
    #     self.assertEqual(type(tensor_string), type(tf.constant([1, 2, 3])), msg=f"data_preprocess.append_start_end_tags did not return tensor string, returned {type(tensor_string)}")
    #     self.assertEqual(tensor_string, "[START] test [END]", msg="data_preprocess.append_start_end_tags returned incorrect string, returned {tensor_string}")

    def test_confirm_process(self):
        import openai_forward.classifier.data_preprocess as data_preprocess
        message_pass = {
            'model': 'gpt-3.5',
            'tools': [
                {
                    'type': 'function',
                    'function': {
                        'name': 'generateImage',
                        'parameters': {
                            'size': 'nan'
                        }
                    }
                }
            ]
        }
        self.assertTrue(data_preprocess.confirm_process_message(message_pass), msg=f"{message_pass} failed to pass")
        message_1 = {
            'model': 'dall-e-2',
            'tools': [
                {
                    'type': 'function',
                    'function': {
                        'name': 'generateImage',
                        'parameters': {
                            'size': 'nan'
                        }
                    }
                }
            ]
        }
        self.assertFalse(data_preprocess.confirm_process_message(message_1), msg=f"{message_1} passed, should fail on model check")
        message_2 = {
            'model': 'gpt-3.5',
            'stools': [
                {
                    'type': 'function',
                    'function': {
                        'name': 'generateImage',
                        'parameters': {
                            'size': 'nan'
                        }
                    }
                }
            ]
        }
        self.assertFalse(data_preprocess.confirm_process_message(message_2), msg=f"{message_2} passed, should fail on tools check")
        message_3 = {
            'model': 'gpt-3.5',
            'stools': [
                {
                    'type': 'const',
                    'function': {
                        'name': 'generateImage',
                        'parameters': {
                            'size': 'nan'
                        }
                    }
                }
            ]
        }
        self.assertFalse(data_preprocess.confirm_process_message(message_3), msg=f"{message_3} passed, should fail on type check")
        message_4 = {
            'model': 'gpt-3.5',
            'stools': [
                {
                    'type': 'function',
                    'function': {
                        'name': 'delImage',
                        'parameters': {
                            'size': 'nan'
                        }
                    }
                }
            ]
        }
        self.assertFalse(data_preprocess.confirm_process_message(message_4), msg=f"{message_4} passed, should fail on name check")
        message_5 = {
            'model': 'gpt-3.5',
            'stools': [
                {
                    'type': 'function',
                    'function': {
                        'name': 'generateImage',
                        'parameters': {
                            'tool_choice': True
                        }
                    }
                }
            ]
        }
        self.assertFalse(data_preprocess.confirm_process_message(message_5), msg=f"{message_5} passed, should fail on tool_choice check")

    def test_preprocess_prompt(self):
        import openai_forward.classifier.data_preprocess as data_preprocess
        data_preprocess.load_vectorizer()        
        tokens = data_preprocess.preprocess_prompt("This is a test prompt string")
        self.assertTrue(tokens[0][0] == 2, msg=f"First token incorrect; [START] == 2, not {tokens[0][0]}")
        self.assertTrue(tokens.shape == (1, 32), msg=f"Token prompt incorrect shape. {tokens.shape} != (1, 32)")

    def test_predict(self):
        # import tensorflow as tf
        import openai_forward.classifier.model as model
        import numpy as np
        prediction = model.predict(np.array([[n for n in range(32)]], dtype=np.int32))
        self.assertEqual(prediction, float(prediction), msg=f"model prediction not float. model predicted {prediction}")
        self.assertTrue(prediction <= 1, msg="Prediciton greater than 1")
        self.assertTrue(prediction >= 0, msg="Prediciton less than 0")

    def test_confidence(self):
        import openai_forward.classifier as classifier
        conf_1 = classifier._convert_confidence(.7)
        conf_2 = classifier._convert_confidence(1 - .2)
        self.assertEqual(conf_1, .4, msg=f"Confidece incorrect, {conf_1} should be .4")
        self.assertEqual(conf_2, .6, msg=f"Confidece incorrect, {conf_2} should be .6")

    def test_prompt(self):
        import openai_forward.classifier as classifier
        response_1 = classifier.classify_prompt({
            'model': 'gpt-3.5',
            'messages': [
                {
                    "content": "If the 'generateImage' function is available, you should never say \"As a text-based AI, I am unable to draw images\", or the like.  Instead, just call the function.",
                    "role": "system"
                },
                {
                    'role': 'user',
                    'content': 'Draw me a picture of an owl'
                }
            ],
            'tools': [
                {
                    'type': 'function',
                    'function': {
                        'name': 'generateImage',
                        'parameters': {
                            'size': 'nan'
                        }
                    }
                }
            ]})
        response_2 = classifier.classify_prompt({
            'model': 'gpt-3.5',
            'messages': [
                {
                    "content": "If the 'generateImage' function is available, you should never say \"As a text-based AI, I am unable to draw images\", or the like.  Instead, just call the function.",
                    "role": "system"
                },
                {
                    'role': 'user',
                    'content': "what is a more creative way of saying, 'job well done'?"
                }
            ],
            'tools': [
                {
                    'type': 'function',
                    'function': {
                        'name': 'generateImage',
                        'parameters': {
                            'size': 'nan'
                        }
                    }
                }
            ]})


if __name__ == '__main__':
    unittest.main()