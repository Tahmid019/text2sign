CONFIG = {
    "DATA_DIR": "./nlp_imp/ISL_Gifs", 
    "GLOSS_MAP": "./nlp_imp/gloss_dataset.json",  # JSON mapping: {"word": ["sample1", ...]}
    "BATCH_SIZE": 16,
    "EPOCHS": 50,
    "MAX_SEQ_LEN": 100,
    "FEATURE_DIM": 63,  # 21 landmarks x 3 coords
    "NUM_CLASSES": None,
    "LR": 1e-4,
    "CHECKPOINT_DIR": "./Train/checkpoints/",
    "LOG_DIR": "./Train/logs/"
}