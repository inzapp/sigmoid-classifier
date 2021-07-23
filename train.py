from sigmoid_classifier import SigmoidClassifier

if __name__ == '__main__':
    SigmoidClassifier(
        train_image_path=r'./train',
        validation_image_path=r'./validation',
        input_shape=(256, 256, 1),
        cycle_lr_params={
            'train_batches': 100010,
            'max_lr': 0.5,
            'min_lr': 0.005,
            'cycle_length': 1000},
        lr=1e-3,
        batch_size=32,
        epochs=300).fit()
