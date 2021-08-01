from sigmoid_classifier import SigmoidClassifier

if __name__ == '__main__':
    SigmoidClassifier(
        train_image_path=r'./train',
        validation_image_path=r'./validation',
        input_shape=(64, 64, 1),
        max_lr=0.5,
        min_lr=0.005,
        burn_in=1000,
        momentum=0.9,
        batch_size=32,
        cycle_length=1000,
        max_batches=300000).fit()
