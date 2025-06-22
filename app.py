import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, Optional
import streamlit as st
import sounddevice as sd
import numpy as np
import wave
import tempfile
import librosa
import whisper
import requests
import pyttsx3
from scipy.stats import skew, kurtosis
from retrying import retry

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration management
class ConfigManager:
    """Manages persistent settings for the application"""
    def __init__(self, config_file="config.json"):
        self.config_file = config_file
        self.config = {
            "voice_id": "21m00Tcm4TlvDq8ikWAM",
            "silence_threshold": 0.01,
            "max_duration": 10,
            "transcription_method": "whisper"
        }
        self.load_config()
    
    def load_config(self):
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    self.config.update(json.load(f))
                logger.info(f"Loaded config from {self.config_file}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
    
    def save_config(self):
        try:
            # Exclude API keys from being saved to the file
            config_to_save = {k: v for k, v in self.config.items() if k not in ["openrouter_api_key", "eleven_labs_api_key"]}
            with open(self.config_file, 'w') as f:
                json.dump(config_to_save, f, indent=2)
            logger.info(f"Saved config to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def get(self, key, default=None):
        return self.config.get(key, default)
    
    def set(self, key, value):
        self.config[key] = value
        self.save_config()

# Voice Input Processor
class ImprovedVoiceInputProcessor:
    """Handles audio recording and transcription"""
    def __init__(self, sample_rate=16000, channels=1, transcription_method="whisper"):
        self.sample_rate = sample_rate
        self.channels = channels
        self.transcription_method = transcription_method
        if transcription_method == "whisper":
            try:
                self.whisper_model = whisper.load_model("tiny")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                st.error(f"❌ Failed to load Whisper model: {e}")
                self.whisper_model = None
        self._test_microphone_setup()
    
    def _test_microphone_setup(self):
        try:
            devices = sd.query_devices()
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            if not input_devices:
                logger.error("No input devices found")
                st.error("❌ No microphone devices found")
                return False
            logger.info(f"Found {len(input_devices)} input device(s)")
            return True
        except Exception as e:
            logger.error(f"Microphone setup error: {e}")
            st.error(f"❌ Microphone setup failed: {e}")
            return False
    
    def _apply_pre_emphasis(self, audio, alpha=0.97):
        return np.append(audio[0], audio[1:] - alpha * audio[:-1])
    
    def _estimate_noise_level(self, audio, window_size=0.1):
        window_samples = int(window_size * self.sample_rate)
        if len(audio) < window_samples:
            return 0.01
        chunks = [audio[i:i + window_samples] for i in range(0, len(audio), window_samples)]
        rms_values = [np.sqrt(np.mean(chunk**2)) for chunk in chunks if len(chunk) == window_samples]
        return np.median(rms_values) if rms_values else 0.01
    
    def record_audio_with_vad(self, max_duration=10, silence_threshold=0.01, silence_duration=2):
        try:
            logger.info("🎤 Starting recording with VAD...")
            buffer_size = int(self.sample_rate * 0.1)
            audio_buffer = []
            silence_counter = 0
            max_silence_chunks = int(silence_duration / 0.1)
            initial_noise_level = None
            
            def audio_callback(indata, frames, time, status):
                if status:
                    logger.warning(f"Audio callback status: {status}")
                processed_audio = self._apply_pre_emphasis(indata.flatten())
                nonlocal initial_noise_level
                if initial_noise_level is None:
                    initial_noise_level = self._estimate_noise_level(processed_audio)
                dynamic_threshold = max(silence_threshold, initial_noise_level * 1.5)
                rms = np.sqrt(np.mean(processed_audio**2))
                audio_buffer.extend(processed_audio)
                nonlocal silence_counter
                if rms > dynamic_threshold:
                    silence_counter = 0
                else:
                    silence_counter += 1
                if (silence_counter >= max_silence_chunks and 
                    len(audio_buffer) > self.sample_rate):
                    raise sd.CallbackStop()
            
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=audio_callback,
                blocksize=buffer_size,
                dtype=np.float32
            ):
                time.sleep(max_duration)
            if not audio_buffer:
                logger.warning("No audio data captured")
                return None
            audio_array = np.array(audio_buffer, dtype=np.float32)
            audio_array = self._trim_silence(audio_array, threshold=silence_threshold)
            logger.info(f"✅ Recorded {len(audio_array)/self.sample_rate:.2f} seconds of audio")
            return audio_array
        except sd.CallbackStop:
            if audio_buffer:
                audio_array = np.array(audio_buffer, dtype=np.float32)
                audio_array = self._trim_silence(audio_array, threshold=silence_threshold)
                logger.info(f"✅ Recording stopped by VAD - {len(audio_array)/self.sample_rate:.2f} seconds")
                return audio_array
            else:
                logger.warning("Recording stopped but no audio captured")
                return None
        except Exception as e:
            logger.error(f"Recording error: {e}")
            return None
    
    def _trim_silence(self, audio, threshold=0.01):
        non_silent = np.where(np.abs(audio) > threshold)[0]
        if len(non_silent) == 0:
            return audio
        start_idx = max(0, non_silent[0] - int(0.1 * self.sample_rate))
        end_idx = min(len(audio), non_silent[-1] + int(0.1 * self.sample_rate))
        return audio[start_idx:end_idx]
    
    def save_audio_to_wav(self, audio_data, filename=None):
        if filename is None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            filename = temp_file.name
            temp_file.close()
        try:
            audio_int16 = (audio_data * 32767).astype(np.int16)
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            logger.info(f"Audio saved to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error saving audio: {e}")
            return None
    
    def transcribe_audio(self, audio_data: np.ndarray) -> Optional[str]:
        if audio_data is None or len(audio_data) == 0:
            return None
        try:
            wav_file = self.save_audio_to_wav(audio_data)
            if not wav_file:
                return None
            if self.transcription_method == "whisper" and self.whisper_model:
                result = self.whisper_model.transcribe(wav_file, language="en")
                os.remove(wav_file)
                return result["text"]
            else:
                logger.warning("No valid transcription method available")
                os.remove(wav_file)
                return ""
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            if os.path.exists(wav_file):
                os.remove(wav_file)
            return None

# Emotion Detector
class EmotionDetector:
    """Extracts comprehensive audio features and provides a textual description"""
    def __init__(self):
        pass
    
    def detect_emotion(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Dict:
        try:
            if len(audio_data) == 0:
                return {"features": {}, "description": "No audio data"}
            features = self._extract_features(audio_data, sample_rate)
            description = self._describe_features(features)
            return {
                "features": features,
                "description": description
            }
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return {"features": {}, "description": "Error extracting features"}
    
    def _extract_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        features = {}
        duration = len(audio_data) / sample_rate
        features['duration'] = float(duration)
        
        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sample_rate)
        pitch_values = pitches[magnitudes > 0]
        if len(pitch_values) > 0:
            features['pitch_mean'] = float(np.mean(pitch_values))
            features['pitch_std'] = float(np.std(pitch_values))
            features['pitch_skew'] = float(skew(pitch_values))
            features['pitch_kurtosis'] = float(kurtosis(pitch_values))
        else:
            features['pitch_mean'] = 0.0
            features['pitch_std'] = 0.0
            features['pitch_skew'] = 0.0
            features['pitch_kurtosis'] = 0.0
        
        rms = librosa.feature.rms(y=audio_data)
        features['energy_mean'] = float(np.mean(rms))
        features['energy_std'] = float(np.std(rms))
        features['energy_skew'] = float(skew(rms.flatten()))
        features['energy_kurtosis'] = float(kurtosis(rms.flatten()))
        
        silence_threshold = 0.01
        silent_frames = rms < silence_threshold
        silence_ratio = float(np.sum(silent_frames) / len(rms))
        features['silence_ratio'] = silence_ratio
        
        tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
        features['tempo'] = float(tempo)
        
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i}_mean'] = float(np.mean(mfcc[i]))
            features[f'mfcc_{i}_std'] = float(np.std(mfcc[i]))
            features[f'mfcc_{i}_skew'] = float(skew(mfcc[i]))
            features[f'mfcc_{i}_kurtosis'] = float(kurtosis(mfcc[i]))
        
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
        features['chroma_mean'] = float(np.mean(chroma))
        features['chroma_std'] = float(np.std(chroma))
        
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
        features['spectral_centroid_std'] = float(np.std(spectral_centroid))
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)
        features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
        features['spectral_bandwidth_std'] = float(np.std(spectral_bandwidth))
        
        spectral_flatness = librosa.feature.spectral_flatness(y=audio_data)
        features['spectral_flatness_mean'] = float(np.mean(spectral_flatness))
        features['spectral_flatness_std'] = float(np.std(spectral_flatness))
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)
        features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
        features['spectral_rolloff_std'] = float(np.std(spectral_rolloff))
        
        zcr = librosa.feature.zero_crossing_rate(audio_data)
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))
        
        ste = librosa.feature.rms(y=audio_data) ** 2
        features['ste_mean'] = float(np.mean(ste))
        features['ste_std'] = float(np.std(ste))
        
        return features
    
    def _describe_features(self, features: Dict) -> str:
        description = "Audio Features:\n"
        description += f"- Duration: {features['duration']:.2f} seconds\n"
        description += f"- Pitch: mean={features['pitch_mean']:.2f} Hz, std={features['pitch_std']:.2f} Hz, skew={features['pitch_skew']:.2f}, kurtosis={features['pitch_kurtosis']:.2f}\n"
        description += f"- Energy: mean={features['energy_mean']:.4f}, std={features['energy_std']:.4f}, skew={features['energy_skew']:.2f}, kurtosis={features['energy_kurtosis']:.2f}\n"
        description += f"- Silence Ratio: {features['silence_ratio']:.2f}\n"
        description += f"- Tempo: {features['tempo']:.2f} BPM\n"
        description += f"- MFCC Means: " + ", ".join([f"mfcc_{i}: {features[f'mfcc_{i}_mean']:.2f}" for i in range(13)]) + "\n"
        description += f"- Chroma: mean={features['chroma_mean']:.4f}, std={features['chroma_std']:.4f}\n"
        description += f"- Spectral Centroid: mean={features['spectral_centroid_mean']:.2f} Hz, std={features['spectral_centroid_std']:.2f} Hz\n"
        description += f"- Spectral Bandwidth: mean={features['spectral_bandwidth_mean']:.2f} Hz, std={features['spectral_bandwidth_std']:.2f} Hz\n"
        description += f"- Spectral Flatness: mean={features['spectral_flatness_mean']:.4f}, std={features['spectral_flatness_std']:.4f}\n"
        description += f"- Spectral Rolloff: mean={features['spectral_rolloff_mean']:.2f} Hz, std={features['spectral_rolloff_std']:.2f} Hz\n"
        description += f"- Zero-Crossing Rate: mean={features['zcr_mean']:.4f}, std={features['zcr_std']:.4f}\n"
        description += f"- Short-Term Energy: mean={features['ste_mean']:.4f}, std={features['ste_std']:.4f}\n"
        if 'speaking_rate' in features:
            description += f"- Speaking Rate: {features['speaking_rate']:.2f} words per minute\n"
        return description

# Response Generator
class ResponseGenerator:
    """Generates empathetic, natural responses using OpenRouter LLM"""
    def __init__(self, api_key: str, model: str = "openai/gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
    
    def generate_response(self, transcribed_text: str, feature_description: str) -> str:
        try:
            system_prompt = f"""You are a supportive friend who understands emotions through words and tone.

Here are examples of how to respond naturally:

- If the user says 'I'm so happy!' with high energy: 'That's awesome! You sound totally thrilled—what's got you so excited?'
- If the user says 'I don't know what to do' with low energy: 'Hey, you sound a bit lost. What's on your mind? I'm here to help.'
- If the user says 'I got a new job!' with excitement: 'No way, that's huge! You sound pumped—tell me all about it!'
- If the user says 'I'm so tired' with a quiet tone: 'You sound exhausted. Rough day? Want to talk it out?'
- If the user says 'I'm fine' with a sad tone: 'You say you're fine, but you sound a bit down. Wanna share what's going on?'
- If the user says 'This is so frustrating!' with a tense tone: 'Ugh, I can tell you're super annoyed. What's got you so worked up?'
- If the user says 'I aced my exam!' with a confident tone: 'That's incredible! You sound so proud—spill the details!'
- If the user says 'I'm not sure about this' with a hesitant tone: 'Sounds like you're feeling a bit unsure. Want to bounce some ideas around?'
- If the user says 'Everything's great!' with a flat tone: 'You're saying it's great, but you don't sound so sure. What's up?'
- If the user says 'I messed up big time' with a shaky tone: 'Oh no, you sound really shaken. What happened? I'm here for you.'

Now, the user said: '{transcribed_text}'. Their voice has these traits: {feature_description}.

Respond in a warm, natural way, reflecting their possible emotional state if it fits. Do NOT mention audio features or analysis. Keep it short (1-3 sentences) and vary your phrasing for a lively feel."""
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": transcribed_text}
                ],
                "max_tokens": 150,
                "temperature": 0.9
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(self.base_url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return "I'm here for you—want to chat about what's going on?"

# Text-to-Speech Engine
class TextToSpeechEngine:
    """Handles text-to-speech conversion using Eleven Labs or pyttsx3"""
    def __init__(self, eleven_labs_api_key=None, voice_id="21m00Tcm4TlvDq8ikWAM"):
        self.eleven_labs_api_key = eleven_labs_api_key
        self.voice_id = voice_id
        self.max_chars = 5000  # Eleven Labs character limit for free tier
        self.last_api_call = 0
        self.min_delay = 1.0  # Minimum delay between API calls to avoid rate limits
        logger.info(f"TextToSpeechEngine initialized with voice_id: {voice_id}, API key provided: {bool(eleven_labs_api_key)}")
    
    def _split_text(self, text: str) -> list:
        """Split text into chunks within character limit"""
        if len(text) <= self.max_chars:
            return [text]
        chunks = []
        current_chunk = ""
        sentences = text.split('. ')
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= self.max_chars:
                current_chunk += sentence + '. '
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + '. '
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks
    
    @retry(stop_max_attempt_number=3, wait_exponential_multiplier=1000, wait_exponential_max=10000)
    def _speak_with_eleven_labs(self, text: str) -> bool:
        try:
            logger.info(f"Attempting to use Eleven Labs API with voice_id: {self.voice_id} for text: {text[:50]}...")
            # Enforce minimum delay to avoid rate limits
            elapsed = time.time() - self.last_api_call
            if elapsed < self.min_delay:
                time.sleep(self.min_delay - elapsed)
            
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
            headers = {
                "xi-api-key": self.eleven_labs_api_key,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg"
            }
            data = {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5
                }
            }
            response = requests.post(url, json=data, headers=headers, timeout=20)
            self.last_api_call = time.time()
            
            if response.status_code == 200:
                # Save audio to temporary file
                temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                temp_audio.write(response.content)
                temp_audio_path = temp_audio.name
                temp_audio.close()
                
                # Verify audio file size
                file_size = os.path.getsize(temp_audio_path)
                logger.info(f"Audio file created: {temp_audio_path}, size: {file_size} bytes")
                
                # Play audio
                with open(temp_audio_path, "rb") as audio_file:
                    st.audio(audio_file, format="audio/mp3", start_time=0)
                
                # Clean up
                os.remove(temp_audio_path)
                logger.info("Successfully played audio with Eleven Labs")
                return True
            else:
                logger.error(f"Eleven Labs API error: {response.status_code} - {response.text}")
                st.error(f"❌ Eleven Labs API error: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error with Eleven Labs API: {str(e)}")
            st.error(f"❌ Error with Eleven Labs API: {str(e)}")
            return False
    
    def _speak_with_pyttsx3(self, text: str):
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            logger.info(f"Speaking with pyttsx3: {text}")
        except Exception as e:
            logger.error(f"pyttsx3 error: {e}")
            st.error(f"❌ pyttsx3 error: Failed to speak the response: {str(e)}")
    
    def speak_text(self, text: str):
        if not text or text.strip() == "":
            logger.warning("No valid text provided for TTS")
            st.error("❌ No text to speak")
            return
        
        logger.info(f"Processing text for TTS: {text}")
        if self.eleven_labs_api_key:
            # Split text into chunks if necessary
            chunks = self._split_text(text)
            success = True
            for i, chunk in enumerate(chunks):
                logger.info(f"Speaking chunk {i+1}/{len(chunks)}: {chunk[:50]}...")
                if not self._speak_with_eleven_labs(chunk):
                    success = False
                    break
            if not success:
                logger.warning("Falling back to pyttsx3 due to Eleven Labs failure")
                st.warning("⚠️ Failed to use Eleven Labs, using robotic voice")
                self._speak_with_pyttsx3(text)
        else:
            logger.warning("No Eleven Labs API key provided, using pyttsx3")
            st.warning("⚠️ No Eleven Labs API key provided, using robotic voice")
            self._speak_with_pyttsx3(text)

# Emotional AI Companion
class EmotionalAICompanion:
    """Main class for live emotional interaction"""
    def __init__(self, openrouter_api_key: str, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.voice_processor = ImprovedVoiceInputProcessor(
            transcription_method=config_manager.get("transcription_method", "whisper")
        )
        self.emotion_detector = EmotionDetector()
        self.response_generator = ResponseGenerator(openrouter_api_key)
        self.tts_engine = TextToSpeechEngine(
            eleven_labs_api_key=config_manager.get("eleven_labs_api_key"),
            voice_id=config_manager.get("voice_id", "21m00Tcm4TlvDq8ikWAM")
        )
        self.conversation_log = []
    
    def process_audio(self, audio_data: np.ndarray, transcribed_text: Optional[str] = None):
        try:
            start_time = time.time()
            emotion_result = self.emotion_detector.detect_emotion(audio_data)
            features = emotion_result["features"]
            
            if transcribed_text and features['duration'] > 0:
                num_words = len(transcribed_text.split())
                speaking_rate = num_words / features['duration'] * 60
                features['speaking_rate'] = float(speaking_rate)
            
            description = self.emotion_detector._describe_features(features)
            
            user_input = transcribed_text or "Audio input processed"
            response_text = self.response_generator.generate_response(user_input, description)
            if response_text:
                logger.info(f"Generated response: {response_text}")
                self.tts_engine.speak_text(response_text)
            else:
                logger.warning("No response text generated, skipping TTS")
                st.error("❌ No response text generated")
                return None
            
            interaction = {
                "timestamp": datetime.now().isoformat(),
                "user_input": user_input,
                "feature_description": description,
                "ai_response": response_text,
                "response_time": time.time() - start_time
            }
            self.conversation_log.append(interaction)
            logger.info(f"Interaction logged - Response time: {interaction['response_time']:.2f}s")
            return interaction
        except Exception as e:
            logger.error(f"Processing error: {e}")
            st.error(f"❌ Processing error: {str(e)}")
            return None

# Custom CSS for UI
def load_css():
    st.markdown("""
    <style>
    /* Hide default Streamlit elements */
    .stApp > header {visibility: hidden;}
    .stApp > footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    .stSidebar {display: none;}
    
    /* Global styling */
    .stApp {
        background: #1a1a1a !important;
        min-height: 100vh;
        color: white;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    }
    
    .main .block-container {
        padding: 1rem;
        max-width: 400px;
        margin: 0 auto;
    }
    
    /* Login page styling */
    .login-container {
        background: #1a1a1a;
        padding: 2rem;
        text-align: center;
        color: white;
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .login-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .app-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .app-title-halo {
        color: white;
    }
    
    .app-title-ai {
        color: #ff0000;
    }
    
    .login-title {
        font-size: 2rem;
        font-weight: 600;
        color: white;
        margin-bottom: 0.5rem;
    }
    
    .login-subtitle {
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    /* Style Streamlit buttons for login */
    .stButton > button {
        background: #2a2a2a !important;
        border: 1px solid #333 !important;
        border-radius: 12px !important;
        color: white !important;
        font-size: 1rem !important;
        height: 60px !important;
        width: 100% !important;
        margin-bottom: 0.8rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background: #333 !important;
        border-color: #444 !important;
        color: white !important;
    }
    
    .stButton > button:focus {
        outline: none !important;
        box-shadow: none !important;
        border-color: #444 !important;
    }
    
    .or-divider {
        text-align: center;
        color: #666;
        margin: 1.5rem 0;
        font-size: 1rem;
        position: relative;
    }
    
    .or-divider::before,
    .or-divider::after {
        content: '';
        position: absolute;
        top: 50%;
        width: 45%;
        height: 1px;
        background: #333;
    }
    
    .or-divider::before {
        left: 0;
    }
    
    .or-divider::after {
        right: 0;
    }
    
    .link-text {
        color: #888;
        font-size: 1rem;
        text-align: center;
        display: block;
        text-decoration: none;
        margin-top: 2rem;
    }
    
    .link-text:hover {
        color: #aaa;
    }
    
    /* Main app styling */
    .main-container {
        background: #1a1a1a;
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        color: white;
        padding: 2rem;
        text-align: center;
    }
    
    .ai-avatar {
        position: relative;
        width: 200px;
        height: 200px;
        margin: 2rem auto;
        border-radius: 50%;
        background: conic-gradient(from 0deg, #8B5CF6, #06B6D4, #10B981, #F59E0B, #EF4444, #8B5CF6);
        display: flex;
        align-items: center;
        justify-content: center;
        animation: rotate 10s linear infinite;
        padding: 8px;
    }
    
    .ai-avatar-inner {
        width: 184px;
        height: 184px;
        border-radius: 50%;
        background: #1a1a1a;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        overflow: hidden;
    }
    
    .ai-avatar-face {
        width: 160px;
        height: 160px;
        border-radius: 50%;
        background: #f0f0f0;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 3rem;
        position: relative;
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Record button for main page */
    .main-container .stButton > button {
        background: #10B981 !important;
        border: none !important;
        border-radius: 50% !important;
        width: 60px !important;
        height: 60px !important;
        font-size: 1.5rem !important;
        color: white !important;
        margin: 2rem auto !important;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.3) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    .main-container .stButton > button:hover {
        background: #059669 !important;
        transform: scale(1.05);
    }
    
    .main-container .stButton > button:disabled {
        background: #555 !important;
        cursor: not-allowed !important;
        transform: none !important;
        box-shadow: none !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        color: white !important;
    }
    
    /* Success/Error messages */
    .stAlert {
        background: rgba(42, 42, 42, 0.8) !important;
        color: white !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
    }
    
    /* Responsive adjustments */
    @media (max-width: 600px) {
        .main .block-container {
            padding: 0.5rem;
        }
        
        .ai-avatar {
            width: 150px;
            height: 150px;
        }
        
        .ai-avatar-inner {
            width: 134px;
            height: 134px;
        }
        
        .ai-avatar-face {
            width: 120px;
            height: 120px;
            font-size: 2rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit UI
def main():
    st.set_page_config(page_title="Halo.ai", page_icon="🤖", layout="wide")
    
    # Load custom CSS
    load_css()
    
    # Initialize ConfigManager
    if 'config_manager' not in st.session_state:
        st.session_state.config_manager = ConfigManager()
    
    # Set API keys from Streamlit secrets
    try:
        st.session_state.config_manager.config["openrouter_api_key"] = st.secrets["openrouter_api_key"]
        st.session_state.config_manager.config["eleven_labs_api_key"] = st.secrets["eleven_labs_api_key"]
    except KeyError as e:
        st.error(f"Missing secret: {e}. Please set the API keys in Streamlit secrets.")
        st.stop()
    
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'login'
    if 'companion' not in st.session_state:
        st.session_state.companion = None
    if 'recording_active' not in st.session_state:
        st.session_state.recording_active = False
    
    # Helper function to handle login
    def handle_login():
        try:
            st.session_state.page = 'main'
            logger.info("Login successful, navigating to main page")
            st.experimental_rerun()
        except Exception as e:
            logger.error(f"Login error: {e}")
            st.error(f"Login failed: {e}")
    
    # Login Page
    if st.session_state.page == 'login':
        # Center the content
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown('<div class="login-header">', unsafe_allow_html=True)
        st.markdown('<h1 class="app-title"><span class="app-title-halo">Halo</span><span class="app-title-ai">.AI</span></h1>', unsafe_allow_html=True)
        st.markdown('<h1 class="login-title">Let\'s Get Started!</h1>', unsafe_allow_html=True)
        st.markdown('<p class="login-subtitle">Discover the latest 1000+ Voice Effects</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Email button
        if st.button("Continue with Email", key="email_login"):
            handle_login()
        
        st.markdown('<div class="or-divider">or</div>', unsafe_allow_html=True)
        
        # Social login buttons
        if st.button("Continue with Google", key="google_login"):
            handle_login()
        
        if st.button("Continue with Apple", key="apple_login"):
            handle_login()
        
        if st.button("Continue with Facebook", key="facebook_login"):
            handle_login()
        
        if st.button("Continue with Twitter", key="twitter_login"):
            handle_login()
        
        st.markdown('<a href="#" class="link-text">Already Have an Account ? Sign in</a>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main Interaction Page
    elif st.session_state.page == 'main':
        try:
            # Initialize components only when needed
            if 'audio_processor' not in st.session_state:
                st.session_state.audio_processor = ImprovedVoiceInputProcessor(
                    transcription_method=st.session_state.config_manager.get("transcription_method", "whisper")
                )
            
            if st.session_state.companion is None:
                with st.spinner("Initializing AI companion..."):
                    st.session_state.companion = EmotionalAICompanion(
                        st.session_state.config_manager.get("openrouter_api_key"),
                        st.session_state.config_manager
                    )
            
            # Render the main page
            st.markdown('<div class="main-container">', unsafe_allow_html=True)
            
            # AI Avatar with animated border
            st.markdown('''
            <div class="ai-avatar">
                <div class="ai-avatar-inner">
                    <div class="ai-avatar-face">
                        👨
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            # Record button
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🎙️", key="record_button", disabled=st.session_state.recording_active):
                    st.session_state.recording_active = True
                    st.experimental_rerun()
            
            if st.session_state.recording_active:
                with st.spinner("Recording... Speak now!"):
                    try:
                        audio_data = st.session_state.audio_processor.record_audio_with_vad(
                            max_duration=st.session_state.config_manager.get("max_duration", 10),
                            silence_threshold=st.session_state.config_manager.get("silence_threshold", 0.01)
                        )
                        st.session_state.recording_active = False
                        
                        if audio_data is not None:
                            transcribed_text = st.session_state.audio_processor.transcribe_audio(audio_data)
                            if transcribed_text:
                                logger.info(f"Transcribed text: {transcribed_text}")
                                interaction = st.session_state.companion.process_audio(audio_data, transcribed_text)
                                if interaction:
                                    logger.info(f"User said: {interaction['user_input']}, AI responded: {interaction['ai_response']}")
                                    st.success(f"You said: {transcribed_text}")
                                    st.info(f"AI: {interaction['ai_response']}")
                                else:
                                    logger.warning("No interaction returned from process_audio")
                                    st.warning("No response generated")
                            else:
                                logger.warning("Transcription returned empty text")
                                st.warning("⚠️ Could not transcribe audio. Please try again.")
                        else:
                            st.error("❌ No audio captured. Check your microphone.")
                        
                        time.sleep(1)
                        st.experimental_rerun()
                        
                    except Exception as e:
                        logger.error(f"Recording error: {e}")
                        st.error(f"Recording failed: {e}")
                        st.session_state.recording_active = False
                        st.experimental_rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            logger.error(f"Main page error: {e}")
            st.error(f"Page loading failed: {e}")
            if st.button("Back to Login"):
                st.session_state.page = 'login'
                st.experimental_rerun()
    
    else:
        # Fallback - should not happen but just in case
        st.error("Unknown page state")
        if st.button("Go to Login"):
            st.session_state.page = 'login'
            st.experimental_rerun()

if __name__ == "__main__":
    main()