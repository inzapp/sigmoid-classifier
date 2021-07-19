from sigmoid_classifier import SigmoidClassifier

if __name__ == '__main__':
    SigmoidClassifier(
        train_image_path=r'./train',
        validation_image_path=r'./validation',
        input_shape=(128, 128, 1),
        lr=1e-3,
        batch_size=32,
        epochs=300).fit()
