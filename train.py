from sigmoid_classifier import SigmoidClassifier

if __name__ == '__main__':
    SigmoidClassifier(
        train_image_path=r'./train',
        validation_image_path=r'./validation',
        input_shape=(128, 128, 1),
        max_lr=0.01,
        min_lr=1e-4,
        cycle_steps=2000,
        batch_size=32,
        epochs=1000).fit()
