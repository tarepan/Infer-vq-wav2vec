<div align="center">

# Infer-vq-wav2vec <!-- omit in toc -->
[![ColabBadge]][notebook]
[![PaperBadge]][paper]  

</div>

Extract vq-wav2vec feature series with only 3 lines of code.

```python
model = torch.hub.load("tarepan/Infer-vq-wav2vec:v1.0.0", "vqw2v_unit", trust_repo=True)
pad   = torch.hub.load("tarepan/Infer-vq-wav2vec:v1.0.0", "vqw2v_pad",  trust_repo=True)

z_series, q_series, q_idx_seriesscore = model(pad(wave))
```

<!-- Auto-generated by "Markdown All in One" extension -->
- [Demo](#demo)
- [How to Use](#how-to-use)
- [References](#references)

## Demo
Extract vq-wav2vec continuous and discrete feature series from your audio:

```python
import torch, torchaudio
import librosa

wave, sr = librosa.load("<your_audio>.wav", sr=None, mono=True)
wave = torchaudio.functional.resample(torch.from_numpy(wave), orig_freq=sr, new_freq=16000).unsqueeze(0)

model = torch.hub.load("tarepan/Infer-vq-wav2vec:v1.0.0", "vqw2v_unit", trust_repo=True)
pad   = torch.hub.load("tarepan/Infer-vq-wav2vec:v1.0.0", "vqw2v_pad",  trust_repo=True)

z_series, q_series, q_idx_seriesscore = model(pad(wave))
```

## How to Use
We use `torch.hub` built-in model loader, so no needs of library import😉  
(As general dependencies, `Python=>3.10` and `torch` are required.)  

First, instantiate a vq-wav2vec model and padding utility:

```python
import torch
model = torch.hub.load("tarepan/Infer-vq-wav2vec:v1.0.0", "vqw2v_unit", trust_repo=True)
pad   = torch.hub.load("tarepan/Infer-vq-wav2vec:v1.0.0", "vqw2v_pad",  trust_repo=True)
```

Then, pass tensor of ***16kHz*** speeches :: (Batch, Time) to the model with padding:

```python
waves_tensor = torch.rand((2, 16000)) # Two speeches, each 1 sec (sr=16,000)
z_series, q_series, q_idx_seriesscore = model(pad(waves_tensor))
```

3 feature series will be returned:

- `z_series` :: (Batch, Time, Feature=512) - Continuous feature series, 100Hz
- `q_series` :: (Batch, Time, Feature=512) - Discrete   feature series, 100Hz
- `q_idx_seriesscore` :: (Batch, Time, Group) - `q_series`'s index series


## References
### Original paper <!-- omit in toc -->
[![PaperBadge]][paper]  
<!-- Generated with the tool -> https://arxiv2bibtex.org/?q=1910.05453&format=bibtex -->
```bibtex
@misc{1910.05453,
Author = {Alexei Baevski and Steffen Schneider and Michael Auli},
Title = {vq-wav2vec: Self-Supervised Learning of Discrete Speech Representations},
Year = {2019},
Eprint = {arXiv:1910.05453},
}
```

### Acknowlegements <!-- omit in toc -->
- [Fairseq](https://github.com/facebookresearch/fairseq)


[ColabBadge]:https://colab.research.google.com/assets/colab-badge.svg

[paper]:https://arxiv.org/abs/1910.05453
[PaperBadge]:https://img.shields.io/badge/paper-arxiv.1910.05453-B31B1B.svg
[notebook]:https://colab.research.google.com/github/tarepan/Infer-vq-wav2vec/blob/main/vqwav2vec.ipynb
[demo page]:https://demo.project.your