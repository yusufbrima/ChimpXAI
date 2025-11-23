import os
from typing import Optional, Tuple, Dict, List
import random
import math
import torchaudio
import torch
from torch.utils.data import Dataset
import torchaudio.transforms as transforms
import torchaudio.transforms as T
from torch import Tensor
from torch.nn import functional as F
import librosa 
import numpy as np
from config import SAMPLING_RATE
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
from torch import Tensor
import random

class Augmentation:
    """
    Applies a set of random audio augmentations to a waveform.
    Uses audiomentations for realistic audio transformations.
    Accepts tunable parameters for hyperparameter search.
    """

    def __init__(
        self,
        sample_rate: int = 41000,
        time_stretch: tuple[float, float, float] = (0.9, 1.1, 0.3),  # min_rate, max_rate, p
        pitch_shift: tuple[int, int, float] = (-2, 2, 0.3),          # min_semitones, max_semitones, p
        shift_p: float = 0.3                                        # probability
    ) -> None:
        self.sample_rate = sample_rate
        min_rate, max_rate, ts_p = time_stretch
        min_semitones, max_semitones, ps_p = pitch_shift

        self.augmenter = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            TimeStretch(min_rate=min_rate, max_rate=max_rate, p=ts_p),
            PitchShift(min_semitones=min_semitones, max_semitones=max_semitones, p=ps_p),
            Shift(p=shift_p),
        ])

    def __call__(self, audio: Tensor) -> Tensor:
        """
        Args:
            audio (Tensor): waveform of shape (1, n_samples)
        
        Returns:
            Tensor: augmented waveform
        """
        audio_np = audio.squeeze(0).numpy()
        # Always apply the augmentations
        audio_np = self.augmenter(samples=audio_np, sample_rate=self.sample_rate)
        return torch.tensor(audio_np, dtype=audio.dtype).unsqueeze(0)

class Noise(torch.nn.Module):
    def __init__(self, min_snr=0.0001, max_snr=0.01):
        """
        :param min_snr: Minimum signal-to-noise ratio
        :param max_snr: Maximum signal-to-noise ratio
        source: https://github.com/Spijkervet/torchaudio-augmentations/blob/master/torchaudio_augmentations/augmentations/noise.py
        """
        super().__init__()
        self.min_snr = min_snr
        self.max_snr = max_snr

    def forward(self, audio):
        std = torch.std(audio)
        noise_std = random.uniform(self.min_snr * std, self.max_snr * std)

        noise = np.random.normal(0.0, noise_std, size=audio.shape).astype(np.float32)

        return audio + noise


class AugmentAudio:
    """
    Applies a set of random audio augmentations to a waveform.
    Includes volume adjustment, frequency masking, and time masking.
    """

    def __init__(self) -> None:
        self.transforms = [
            T.Vol(gain=0.9),  # Reduce volume by 10%
            T.FrequencyMasking(freq_mask_param=15),  # Mask random frequencies
            T.TimeMasking(time_mask_param=35)  # Mask random time intervals
        ]

    def __call__(self, audio: Tensor) -> Tensor:
        for transform in self.transforms:
            if random.random() > 0.5:
                audio = transform(audio)
        return audio


class AudioDataset(Dataset):
    """
    A PyTorch Dataset class for loading audio data from a folder structure where each subdirectory is a class.

    Attributes:
        root_dir (str): The root directory of the dataset.
        classes (list): List of class names (subdirectory names).
        file_list (list): List of (file_path, class_idx) tuples.
        transform (callable, optional): Optional transform to be applied on a sample.
        duration (float): The duration (in seconds) of audio samples to extract.
        sample_rate (int): The sample rate of the audio files.
    """

    def __init__(self, root_dir, duration=5, transform=None):
        """
        Initialize the dataset with the root directory.

        Parameters:
            root_dir (str): Path to the root directory containing class subdirectories.
            duration (float, optional): Duration (in seconds) of audio samples to extract. Default is 5 seconds.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.duration = duration
        self.transform = transform
        
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.classes.sort()  # Ensure consistent class ordering
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        self.file_list = []
        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            for filename in os.listdir(class_path):
                if filename.endswith(('.wav', '.mp3', '.flac')):  # Add more audio formats if needed
                    self.file_list.append((os.path.join(class_path, filename), self.class_to_idx[class_name]))
        
        # Determine the sample rate from the first audio file
        first_audio_path = self.file_list[0][0]
        waveform, sample_rate = torchaudio.load(first_audio_path)
        self.sample_rate = sample_rate

    def get_class_name(self, class_idx):
        """
        Returns the class name corresponding to the given class index.

        Parameters:
            class_idx (int): The class index.

        Returns:
            str: The corresponding class name.
        """
        for name, idx in self.class_to_idx.items():
            if idx == class_idx:
                return name
        return None  # Return None if the class index is not found

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Generates one sample of data.

        Parameters:
            idx (int): Index of the sample to be fetched.

        Returns:
            tuple: A tuple containing the audio sample (waveform and sample rate) and the label (class ID).
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the file path and label
        audio_path, label_idx = self.file_list[idx]

        # Load the audio file
        waveform, _ = torchaudio.load(audio_path)

        # Apply transform if provided
        if self.transform:
            waveform = self.transform(waveform)

        # Calculate the number of samples to extract
        samples_to_extract = int(self.duration * self.sample_rate)

        # Pad or truncate the waveform to the desired duration
        if waveform.size(1) < samples_to_extract:
            pad_length = samples_to_extract - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_length), 'constant', 0)
        elif waveform.size(1) > samples_to_extract:
            waveform = waveform[:, :samples_to_extract]

        sample = {'data': waveform, 'sample_rate': self.sample_rate}

        return sample, label_idx

class MAudioDataset(Dataset):
    """
    A PyTorch Dataset class for loading audio data from a folder structure where each subdirectory is a class.
    This version splits longer audio samples into segments of the desired duration, pads shorter segments,
    and resamples all audio to a specified sample rate.

    Attributes:
        root_dir (str): The root directory of the dataset.
        classes (list): List of class names (subdirectory names).
        file_list (list): List of (file_path, class_idx, start_time) tuples.
        transform (callable, optional): Optional transform to be applied on a sample.
        duration (float): The duration (in seconds) of audio samples to extract.
        target_sample_rate (int): The target sample rate for all audio files.
    """

    def __init__(self, root_dir, duration=5, target_sample_rate=SAMPLING_RATE, transform=None):
        """
        Initialize the dataset with the root directory.

        Parameters:
            root_dir (str): Path to the root directory containing class subdirectories.
            duration (float, optional): Duration (in seconds) of audio samples to extract. Default is 5 seconds.
            target_sample_rate (int, optional): Target sample rate for all audio. Default is 44100 Hz.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.duration = duration
        self.target_sample_rate = target_sample_rate
        self.transform = transform
        
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.classes.sort()  # Ensure consistent class ordering
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        self.file_list = []
        self.file_sample_rates = []
        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            for filename in os.listdir(class_path):
                if filename.endswith(('.wav', '.mp3', '.flac')):  # Add more audio formats if needed
                    file_path = os.path.join(class_path, filename)
                    self._add_file_segments(file_path, self.class_to_idx[class_name])
        
        if not self.file_list:
            raise ValueError("No audio files found in the specified directory.")
    
    def _add_file_segments(self, file_path, class_idx):
        """
        Add file segments to the file_list based on audio duration.
        Uses librosa to get audio metadata (no torchaudio needed).

        Parameters:
            file_path (str): Path to the audio file.
            class_idx (int): Class index for the audio file.
        """
        # Load only the metadata (duration and sample rate)
        y, sr = librosa.load(file_path, sr=None, mono=True)
        audio_length = len(y) / sr
        num_segments = math.ceil(audio_length / self.duration)

        for i in range(num_segments):
            start_time = i * self.duration
            self.file_list.append((file_path, class_idx, start_time))
            self.file_sample_rates.append(sr)
    # def _add_file_segments(self, file_path, class_idx):
    #     """
    #     Add file segments to the file_list based on audio duration.

    #     Parameters:
    #         file_path (str): Path to the audio file.
    #         class_idx (int): Class index for the audio file.
    #     """
    #     audio_metadata = torchaudio.info(file_path)

    #     audio_length = audio_metadata.num_frames / audio_metadata.sample_rate
    #     num_segments = math.ceil(audio_length / self.duration)
        
    #     for i in range(num_segments):
    #         start_time = i * self.duration
    #         self.file_list.append((file_path, class_idx, start_time))
    #         self.file_sample_rates.append(audio_metadata.sample_rate)

    def get_class_name(self, class_idx):
        """
        Returns the class name corresponding to the given class index.

        Parameters:
            class_idx (int): The class index.

        Returns:
            str: The corresponding class name.
        """
        for name, idx in self.class_to_idx.items():
            if idx == class_idx:
                return name
        return None  # Return None if the class index is not found

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Generates one sample of data using librosa instead of torchaudio.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get file path, label, start time
        audio_path, label_idx, start_time = self.file_list[idx]
        sr = self.file_sample_rates[idx]

        # Load the full audio with librosa
        y, file_sr = librosa.load(audio_path, sr=None, mono=True)

        # Compute start and end sample indices for the segment
        start_sample = int(start_time * file_sr)
        end_sample = start_sample + int(self.duration * file_sr)
        end_sample = min(end_sample, len(y))  # avoid overflow

        # Extract the segment
        segment = y[start_sample:end_sample]

        # Resample if needed
        if file_sr != self.target_sample_rate:
            segment = librosa.resample(segment, orig_sr=file_sr, target_sr=self.target_sample_rate)

        # Convert to (channels, samples) to mimic torchaudio
        waveform = torch.tensor(segment, dtype=torch.float32).unsqueeze(0)

        # Pad or truncate to exactly self.duration
        samples_to_extract = int(self.duration * self.target_sample_rate)
        if waveform.size(1) < samples_to_extract:
            pad_length = samples_to_extract - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        elif waveform.size(1) > samples_to_extract:
            waveform = waveform[:, :samples_to_extract]

        # Apply transform if provided
        if self.transform:
            waveform = self.transform(waveform)

        sample = {'data': waveform, 'sample_rate': self.target_sample_rate}

        return sample, label_idx


class SpectrogramDataset(MAudioDataset):
    """
    A PyTorch Dataset class for loading audio data and converting it to log mel spectrograms.
    This class inherits from MAudioDataset and adds spectrogram transformation.

    Attributes:
        n_mels (int): Number of mel filterbanks.
        n_fft (int): Size of FFT.
        hop_length (int): Number of samples between successive frames.
        power (float): Exponent for the magnitude spectrogram.
        normalize (bool): Whether to normalize the spectrograms.
    """

    def __init__(self, root_dir, duration=5, target_sample_rate=SAMPLING_RATE, n_fft=512, 
                 hop_length=256, power=2.0, normalize=True, transform=None):
        """
        Initialize the dataset with the root directory and spectrogram parameters.

        Parameters:
            root_dir (str): Path to the root directory containing class subdirectories.
            duration (float, optional): Duration (in seconds) of audio samples to extract. Default is 5 seconds.
            target_sample_rate (int, optional): Target sample rate for all audio. Default is 44100 Hz.
            n_fft (int, optional): Size of FFT. Default is 512.
            hop_length (int, optional): Number of samples between successive frames. Default is 256.
            power (float, optional): Exponent for the magnitude spectrogram. Default is 2.0.
            normalize (bool, optional): Whether to normalize the spectrograms. Default is True.
            transform (callable, optional): Optional transform to be applied on the spectrogram.
        """
        super().__init__(root_dir, duration, target_sample_rate, transform)
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.power = power
        self.normalize = normalize
        
        self.spectrogram = transforms.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length, power=self.power)
        # log_spectrogram = transforms.AmplitudeToDB()(spectrogram)

        # self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        #     sample_rate=target_sample_rate,
        #     n_fft=n_fft,
        #     hop_length=hop_length,
        #     power=power
        # )

    def __getitem__(self, idx):
        """
        Generates one sample of data.

        Parameters:
            idx (int): Index of the sample to be fetched.

        Returns:
            tuple: A tuple containing the log mel spectrogram and the label (class ID).
        """
        sample, label_idx = super().__getitem__(idx)
        waveform = sample['data']

        # Convert to mono if stereo
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Compute mel spectrogram
        spectrogram = self.spectrogram(waveform)

        # Convert to decibels
        log_spectrogram = torchaudio.transforms.AmplitudeToDB()(spectrogram)

        # Normalize if required
        if self.normalize:
            log_spectrogram = (log_spectrogram - log_spectrogram.mean()) / log_spectrogram.std()

        # Apply additional transform if provided
        if self.transform:
            log_spectrogram = self.transform(log_spectrogram)
        
        sample = {'data': log_spectrogram,'waveform': waveform, 'sample_rate': self.target_sample_rate}

        return sample, label_idx

class AugAudioDataset(Dataset):
    """
    A PyTorch Dataset for loading and segmenting audio files from a directory structure
    where each subdirectory represents a class.

    Args:
        root_dir (str): Root directory containing class-named subdirectories with audio files.
        duration (float): Duration (in seconds) for each audio segment.
        target_sample_rate (int): Desired sample rate for all audio.
        transform (callable, optional): Optional waveform transform (e.g., augmentations).
    """

    def __init__(
        self,
        root_dir: str,
        duration: float = 5.0,
        target_sample_rate: int = SAMPLING_RATE,
        transform: Optional[callable] = None
    ) -> None:
        self.root_dir = root_dir
        self.duration = duration
        self.target_sample_rate = target_sample_rate
        self.transform = transform

        # Collect classes and map to integer labels
        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Generate list of (file_path, label_idx, start_time)
        self.file_list: List[Tuple[str, int, float]] = []
        self.file_sample_rates: List[int] = []

        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            for filename in os.listdir(class_path):
                if filename.endswith(('.wav', '.mp3', '.flac')):
                    file_path = os.path.join(class_path, filename)
                    self._add_file_segments(file_path, self.class_to_idx[class_name])

        if not self.file_list:
            raise ValueError("No audio files found in the specified directory.")

    def _add_file_segments(self, file_path: str, class_idx: int) -> None:
        """
        Splits a long audio file into fixed-duration segments and adds them to the dataset.

        Args:
            file_path (str): Path to the audio file.
            class_idx (int): Integer label of the class.
        """
        # Load the audio header only (librosa can provide sr without loading full audio)
        # librosa.get_duration can read duration without loading entire audio
        y, sr = librosa.load(file_path, sr=None, mono=True)
        audio_length = len(y) / sr  # duration in seconds

        num_segments = math.ceil(audio_length / self.duration)

        for i in range(num_segments):
            start_time = i * self.duration
            self.file_list.append((file_path, class_idx, start_time))
            self.file_sample_rates.append(sr)


    def __len__(self) -> int:
        return len(self.file_list)



    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        Loads and returns an audio waveform segment and its label.

        Returns:
            Dict with waveform and sample rate, and the integer label.
        """
        audio_path, label_idx, start_time = self.file_list[idx]
        sr = self.file_sample_rates[idx]

        # --- Load segment using librosa ---
        waveform_np, sample_rate = librosa.load(
            audio_path,
            sr=sr,                  # preserve original sample rate
            mono=True,              # convert to mono
            offset=start_time,      # start time in seconds
            duration=self.duration  # duration in seconds
        )

        # Convert to tensor and shape [channels, samples] like torchaudio
        waveform = torch.tensor(waveform_np).unsqueeze(0)  # [1, num_samples]

        # --- Resample if needed ---
        if sample_rate != self.target_sample_rate:
            waveform = torch.nn.functional.interpolate(
                waveform.unsqueeze(0),  # add batch dim
                size=int(waveform.size(1) * self.target_sample_rate / sample_rate),
                mode='linear',
                align_corners=False
            ).squeeze(0)
            sample_rate = self.target_sample_rate

        # --- Pad or truncate to match the desired length ---
        samples_to_extract = int(self.duration * self.target_sample_rate)
        if waveform.size(1) < samples_to_extract:
            pad_length = samples_to_extract - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_length), mode='constant', value=0)
        elif waveform.size(1) > samples_to_extract:
            waveform = waveform[:, :samples_to_extract]

        # --- Apply optional waveform transformation (augmentation) ---
        if self.transform:
            waveform = self.transform(waveform)

        return {'data': waveform, 'sample_rate': self.target_sample_rate}, label_idx


class AugSpectrogramDataset(AugAudioDataset):
    """
    Dataset subclass that returns log-scaled spectrograms instead of raw waveforms.

    Args:
        n_fft (int): FFT window size.
        hop_length (int): Hop length for STFT.
        power (float): Power of the spectrogram (e.g., 2 for power, 1 for magnitude).
        normalize (bool): Whether to normalize the spectrogram.
    """

    def __init__(
        self,
        root_dir: str,
        duration: float = 5.0,
        target_sample_rate: int = SAMPLING_RATE,
        n_fft: int = 512,
        hop_length: int = 256,
        power: float = 2.0,
        normalize: bool = True,
        transform: Optional[callable] = None
    ) -> None:
        super().__init__(root_dir, duration, target_sample_rate, transform)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.power = power
        self.normalize = normalize

        self.spectrogram = T.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length, power=self.power)
        self.db_transform = T.AmplitudeToDB()

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Tensor], int]:
        """
        Returns:
            Dict with 'data' (log-scaled spectrogram), 'waveform', and 'sample_rate',
            and the label index.
        """
        sample, label_idx = super().__getitem__(idx)
        waveform = sample['data']

        # Convert to mono if stereo
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Convert waveform to log-spectrogram
        spectrogram = self.spectrogram(waveform)
        log_spectrogram = self.db_transform(spectrogram)

        # Normalize spectrogram
        if self.normalize:
            log_spectrogram = (log_spectrogram - log_spectrogram.mean()) / (log_spectrogram.std() + 1e-5)

        return {
            'data': log_spectrogram,
            'waveform': waveform,
            'sample_rate': self.target_sample_rate
        }, label_idx




if __name__ == "__main__":
    pass
