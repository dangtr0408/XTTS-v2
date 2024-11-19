import os
import soundfile as sf
import torch

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


if __name__ == "__main__":
    # Load the text from the file
    with open('text.txt', 'r', encoding='utf-8') as file:
        text = file.read()

    config = XttsConfig()
    config.load_json("./weights/config.json")
    XTTS_MODEL = Xtts.init_from_config(config)
    XTTS_MODEL.load_checkpoint(config, checkpoint_dir="./weights/")
    XTTS_MODEL.eval()
    if torch.cuda.is_available():
        XTTS_MODEL.cuda()

    gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
        audio_path="./weights/rapper.wav",
        gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
        max_ref_length=XTTS_MODEL.config.max_ref_len,
        sound_norm_refs=XTTS_MODEL.config.sound_norm_refs,
    )

    out_wav = XTTS_MODEL.inference(
        text=text,
        language="vi",
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=0.3,
        length_penalty=10.0,
        repetition_penalty=5.0,
        top_k=30,
        top_p=0.85,
    )

    sf.write('output1.wav', out_wav['wav'], 24000)
