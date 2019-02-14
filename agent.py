import torch as th

from models.classifiers import RGBClassifier, DepthClassifier, GreyscaleClassifier

class Agent(object):

    def __init__(self):
        self.rgb_classifier = RGBClassifier()
        self.depth_classifier = DepthClassifier()
        self.greyscale_classifier = GreyscaleClassifier()
        self.classifiers = {'rgb': self.rgb_classifier,
                            'depth': self.depth_classifier,
                            'greyscale': self.greyscale_classifier
                            }

    def predict(self, input_boxes, classifier_type='rgb', use_cuda=True):
        model = self.classifiers[classifier_type]
        
        input_boxes = th.Tensor(input_boxes)
        if use_cuda:
            input_boxes = input_boxes.cuda()
        sigmoid_probs = model(input_boxes)
        return sigmoid_probs

    def gradient_step()

        