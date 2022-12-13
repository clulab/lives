
import torch
import whisper
import numpy as np


def set_device():
    # gpu or cpu
    torch.cuda.is_available()
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(device, model_type="base"):
    # load the model
    # we are using model type 'base' so far
    model = whisper.load_model(model_type, device=device)
    print(
        f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
        f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
    )

    return model


def save_transcriptions(df_to_save, savename):
    df_to_save.to_csv(f"./output/{savename}", index=False)