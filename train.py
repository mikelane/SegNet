from pathlib import Path

from keras.callbacks import TensorBoard, ModelCheckpoint

from data import unet_flow_from_csv
from network import Network

if __name__ == '__main__':
    img_rows = 512
    img_cols = 512
    img_channels = 3
    batch_size = 3
    num_epochs = 100
    network_channel_sizes = (8, 16, 32, 64, 128)
    unet_csv_path = Path('TrainingImagesMaskSpeckles/train_test_filepath.csv').resolve()
    model_name = 'NextasSegNet'
    model_base_path = Path('models/').resolve()

    train_generator, validation_generator, num_training_samples, num_validation_samples = unet_flow_from_csv(filename=str(unet_csv_path),
                                                                                                             batch_size=batch_size,
                                                                                                             img_shape=(img_rows, img_cols, img_channels))

    unet = Network(input_width=img_cols,
                   input_height=img_rows,
                   input_channels=img_channels,
                   network_channel_sizes=network_channel_sizes,
                   leaky_alpha=0.1)
    model = unet.get_model()

    with open(model_base_path / f'{model_name}.yaml', 'w') as f:
        f.write(model.to_yaml())

    for epoch in range(num_epochs):
        # TODO: implement logging
        print(f'Beginning epoch {epoch}\n')

        epoch_model_path = model_base_path / f'{model_name}_epoch_{epoch}'

        model.fit_generator(genrator=train_generator,
                            steps_per_epoch=num_training_samples // batch_size,
                            validation_data=validation_generator,
                            validation_steps=num_validation_samples // batch_size,
                            callbacks=[TensorBoard(log_dir='logs/'),
                                       ModelCheckpoint(f'{epoch_model_path}.h5',
                                                       monitor='loss',
                                                       save_best_only=False,
                                                       save_weights_only=True,
                                                       mode='auto',
                                                       period=1)])
