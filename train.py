"""
Authors : inzapp

Github url : https://github.com/inzapp/sigmoid-classifier

Copyright 2021 inzapp Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License"),
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import argparse

from sigmoid_classifier import SigmoidClassifier


if __name__ == '__main__':
    classifier = SigmoidClassifier(
        train_image_path=r'/train_data/imagenet/train',
        validation_image_path=r'/train_data/imagenet/validation',
        input_shape=(64, 64, 1),
        lr=0.001,
        alpha=0.5,
        gamma=2.0,
        warm_up=0.5,
        momentum=0.9,
        batch_size=32,
        iterations=1000000,
        label_smoothing=0.1,
        aug_brightness=0.3,
        aug_contrast=0.4,
        aug_rotate=20,
        aug_h_flip=False,
        checkpoint_interval=20000,
        show_class_activation_map=False)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='', help='pretrained model path')
    parser.add_argument('--evaluate', action='store_true', help='evaluate using train or validation dataset')
    parser.add_argument('--dataset', type=str, default='validation', help='dataset for evaluate, train or validation available')
    args = parser.parse_args()
    if args.model != '':
        classifier.load_model(args.model)
    if args.evaluate:
        classifier.evaluate(dataset=args.dataset)
    else:
        classifier.train()

