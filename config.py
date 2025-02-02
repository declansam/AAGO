class Config:
    GRID_SIZE = 50 * 50
    WEATHER_FEATURES = 3
    SEQUENCE_LENGTH = 4
    BATCH_SIZE = 1024  # Reduced batch size
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3  # Modified learning rate
    D_MODEL = 256  # Reduced model dimension
    N_HEAD = 8
    N_LAYERS = 6
    DROPOUT = 0.2