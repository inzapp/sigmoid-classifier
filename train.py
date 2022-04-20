from sigmoid_classifier import SigmoidClassifier

if __name__ == '__main__':
    SigmoidClassifier(
        train_image_path=r'/train_data/imagenet/train',
        validation_image_path=r'/train_data/imagenet/validation',
        input_shape=(224, 224, 1),
        lr=0.001,
        decay=0.0005,
        momentum=0.9,
        batch_size=32,
        iterations=1000000).fit()
