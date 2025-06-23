import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, Optional
import streamlit as st
import numpy as np
import wave
import tempfile
import librosa
import whisper
import requests
import pyttsx3
from scipy.stats import skew, kurtosis
from retrying import retry
from io import BytesIO
import base64
from pydub import AudioSegment

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
            "silence_threshold": 0.005,
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
            config_to_save = {k: v for k, v in self.config.items() if k not in ["openrouter_api_key", "eleven_labs_api_key"]}
            os.makedirs(os.path.dirname(os.path.abspath(self.config_file)), exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(config_to_save, f, indent=2)
            logger.info(f"Saved config to {self.config_file}")
        except PermissionError:
            logger.warning(f"Permission denied writing to {self.config_file}")
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
    _whisper_model = None  # Class-level cache to avoid reloading

    def __init__(self, sample_rate=16000, channels=1, transcription_method="whisper"):
        self.sample_rate = sample_rate
        self.channels = channels
        self.transcription_method = transcription_method
        if transcription_method == "whisper":
            if ImprovedVoiceInputProcessor._whisper_model is None:
                try:
                    ImprovedVoiceInputProcessor._whisper_model = whisper.load_model("base")
                    logger.info("Whisper model 'base' loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load Whisper model: {e}")
                    st.error(f"❌ Failed to load Whisper model: {e}")
            self.whisper_model = ImprovedVoiceInputProcessor._whisper_model
    
    def process_webm_audio(self, audio_file):
        """Process WebM audio data from browser recording"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_file:
                audio_bytes = audio_file.read()
                logger.info(f"Audio file size: {len(audio_bytes)} bytes")
                temp_file.write(audio_bytes)
                temp_file_path = temp_file.name
            logger.info(f"Temp file created: {temp_file_path}")
            wav_path = temp_file_path.replace('.webm', '.wav')
            AudioSegment.from_file(temp_file_path, format='webm').export(wav_path, format='wav')
            audio_data, sr = librosa.load(wav_path, sr=self.sample_rate, mono=True)
            logger.info(f"Audio loaded: {len(audio_data)} samples, sample rate: {sr}")
            os.remove(temp_file_path)
            os.remove(wav_path)
            audio_data = self._trim_silence(audio_data)
            duration = len(audio_data) / self.sample_rate
            logger.info(f"✅ Processed audio - {duration:.2f} seconds")
            if duration < 0.5:
                logger.warning("Audio duration too short (<0.5s), may affect transcription")
                st.warning("⚠️ Audio too short. Please speak for at least 1 second.")
                return None
            return audio_data
        except Exception as e:
            logger.error(f"Error processing WebM audio: {e}")
            st.error(f"❌ Error processing audio: {e}")
            return None
    
    def _trim_silence(self, audio, threshold=0.005):
        """Trim silence from audio data"""
        if len(audio) == 0:
            logger.warning("Empty audio data received")
            return audio
        non_silent = np.where(np.abs(audio) > threshold)[0]
        if len(non_silent) == 0:
            logger.warning("No non-silent audio detected")
            return audio
        start_idx = max(0, non_silent[0] - int(0.1 * self.sample_rate))
        end_idx = min(len(audio), non_silent[-1] + int(0.1 * self.sample_rate))
        return audio[start_idx:end_idx]
    
    def save_audio_to_wav(self, audio_data, filename=None):
        """Save audio data to WAV file"""
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
        """Transcribe audio data using Whisper"""
        if audio_data is None or len(audio_data) == 0:
            logger.warning("No audio data for transcription")
            return None
        try:
            wav_file = self.save_audio_to_wav(audio_data)
            if not wav_file:
                logger.error("Failed to save audio to WAV")
                return None
            if self.transcription_method == "whisper" and self.whisper_model:
                result = self.whisper_model.transcribe(wav_file, language="en")
                transcribed_text = result["text"].strip()
                logger.info(f"Transcription result: {transcribed_text}")
                os.remove(wav_file)
                return transcribed_text if transcribed_text else None
            else:
                logger.warning("No valid transcription method available")
                os.remove(wav_file)
                return None
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
                logger.warning("Empty audio data received for emotion detection")
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
        
        silence_threshold = 0.005
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
        description += f"- Duration: {features.get('duration', 0):.2f} seconds\n"
        description += f"- Pitch: mean={features.get('pitch_mean', 0):.2f} Hz, std={features.get('pitch_std', 0):.2f} Hz, skew={features.get('pitch_skew', 0):.2f}, kurtosis={features.get('pitch_kurtosis', 0):.2f}\n"
        description += f"- Energy: mean={features.get('energy_mean', 0):.4f}, std={features.get('energy_std', 0):.4f}, skew={features.get('energy_skew', 0):.2f}, kurtosis={features.get('energy_kurtosis', 0):.2f}\n"
        description += f"- Silence Ratio: {features.get('silence_ratio', 0):.2f}\n"
        description += f"- Tempo: {features.get('tempo', 0):.2f} BPM\n"
        description += f"- MFCC Means: " + ", ".join([f"mfcc_{i}: {features.get(f'mfcc_{i}_mean', 0):.2f}" for i in range(13)]) + "\n"
        description += f"- Chroma: mean={features.get('chroma_mean', 0):.4f}, std={features.get('chroma_std', 0):.4f}\n"
        description += f"- Spectral Centroid: mean={features.get('spectral_centroid_mean', 0):.2f} Hz, std={features.get('spectral_centroid_std', 0):.2f} Hz\n"
        description += f"- Spectral Bandwidth: mean={features.get('spectral_bandwidth_mean', 0):.2f} Hz, std={features.get('spectral_bandwidth_std', 0):.2f} Hz\n"
        description += f"- Spectral Flatness: mean={features.get('spectral_flatness_mean', 0):.4f}, std={features.get('spectral_flatness_std', 0):.4f}\n"
        description += f"- Spectral Rolloff: mean={features.get('spectral_rolloff_mean', 0):.2f} Hz, std={features.get('spectral_rolloff_std', 0):.2f} Hz\n"
        description += f"- Zero-Crossing Rate: mean={features.get('zcr_mean', 0):.4f}, std={features.get('zcr_std', 0):.4f}\n"
        description += f"- Short-Term Energy: mean={features.get('ste_mean', 0):.4f}, std={features.get('ste_std', 0):.4f}\n"
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
            
            logger.info("Sending request to OpenRouter API")
            response = requests.post(self.base_url, json=payload, headers=headers, timeout=20)
            logger.info(f"OpenRouter response status: {response.status_code}")
            response.raise_for_status()
            
            result = response.json()
            response_text = result["choices"][0]["message"]["content"].strip()
            logger.info(f"Generated response: {response_text}")
            return response_text
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            st.error(f"❌ Response generation failed: {e}")
            return "I'm here for you—want to chat about what's going on?"

# Text-to-Speech Engine
class TextToSpeechEngine:
    """Handles text-to-speech conversion using Eleven Labs or pyttsx3"""
    def __init__(self, eleven_labs_api_key=None, voice_id="21m00Tcm4TlvDq8ikWAM"):
        self.eleven_labs_api_key = eleven_labs_api_key
        self.voice_id = voice_id
        self.max_chars = 5000
        self.last_api_call = 0
        self.min_delay = 1.0
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
            logger.info(f"Attempting to use Eleven Labs API for text: {text[:50]}...")
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
                temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                temp_audio.write(response.content)
                temp_audio_path = temp_audio.name
                temp_audio.close()
                
                file_size = os.path.getsize(temp_audio_path)
                logger.info(f"Audio file created: {temp_audio_path}, size: {file_size} bytes")
                
                with open(temp_audio_path, "rb") as audio_file:
                    st.audio(audio_file, format="audio/mp3", start_time=0)
                
                try:
                    if os.path.exists(temp_audio_path):
                        os.remove(temp_audio_path)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp audio file: {e}")
                
                logger.info("Successfully played audio with Eleven Labs")
                return True
            else:
                logger.error(f"Eleven Labs API error: {response.status_code} - {response.text}")
                st.error(f"❌ Eleven Labs API error: {response.status_code}")
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
            
            if not features:
                logger.error("No features extracted from audio")
                return None
            
            if transcribed_text and features.get('duration', 0) > 0:
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
    
    /* Audio recording button */
    #recordButton, #stopButton {
        background: #10B981;
        border: none;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        font-size: 1.5rem;
        color: white;
        margin: 1rem;
        cursor: pointer;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.3);
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    #recordButton:hover, #stopButton:hover {
        background: #059669;
        transform: scale(1.05);
    }
    
    #recordButton:disabled, #stopButton:disabled {
        background: #555;
        cursor: not-allowed;
        transform: none;
        box-shadow: none;
    }
    
    /* Status message */
    #status {
        color: white;
        font-size: 1rem;
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# JavaScript for Audio Recording
def audio_recorder_component():
    """Embed JavaScript for browser-based audio recording"""
    component_html = """
    <div style="text-align: center;">
        <button id="recordButton">🎙️</button>
        <button id="stopButton" disabled>🛑</button>
        <div id="status">Press the microphone to start recording</div>
        <input type="file" id="audioFile" accept="audio/*" style="display: none;">
    </div>
    <script>
        let mediaRecorder;
        let audioChunks = [];
        const recordButton = document.getElementById('recordButton');
        const stopButton = document.getElementById('stopButton');
        const status = document.getElementById('status');
        const audioFileInput = document.getElementById('audioFile');

        async function startRecording() {
            try {
                console.log('Requesting microphone access...');
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                console.log('Microphone access granted');
                mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
                audioChunks = [];
                
                mediaRecorder.ondataavailable = event => {
                    if (event.data.size > 0) {
                        audioChunks.push(event.data);
                        console.log('Audio chunk received, size:', event.data.size);
                    }
                };
                
                mediaRecorder.onstop = () => {
                    console.log('Recording stopped, chunks:', audioChunks.length);
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    console.log('Audio blob created, size:', audioBlob.size);
                    if (audioBlob.size === 0) {
                        console.error('Audio blob is empty');
                        status.textContent = 'Error: No audio recorded';
                        return;
                    }
                    const reader = new FileReader();
                    reader.onload = () => {
                        console.log('Base64 audio data ready, length:', reader.result.length);
                        window.parent.postMessage({
                            type: 'streamlit:setComponentValue',
                            value: reader.result
                        }, '*');
                    };
                    reader.readAsDataURL(audioBlob);
                    stream.getTracks().forEach(track => track.stop());
                    console.log('Audio stream stopped');
                };
                
                mediaRecorder.start();
                recordButton.disabled = true;
                stopButton.disabled = false;
                status.textContent = 'Recording... Speak now!';
                console.log('Recording started');
            } catch (err) {
                console.error('Error accessing microphone:', err);
                status.textContent = 'Error accessing microphone: ' + err.message;
            }
        }

        function stopRecording() {
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
                recordButton.disabled = false;
                stopButton.disabled = true;
                status.textContent = 'Processing audio...';
                console.log('Recording stopped by user');
            }
        }

        recordButton.addEventListener('click', startRecording);
        stopButton.addEventListener('click', stopRecording);
    </script>
    """
    return st.components.v1.html(component_html, height=200)

# Streamlit UI
def main():
    st.set_page_config(page_title="Halo.ai", page_icon="🤖", layout="wide")
    
    load_css()
    
    if 'config_manager' not in st.session_state:
        st.session_state.config_manager = ConfigManager()
    
    try:
        openrouter_key = st.secrets.get("openrouter_api_key")
        eleven_labs_key = st.secrets.get("eleven_labs_api_key")
        if not openrouter_key:
            st.error("OpenRouter API key not found in secrets")
            st.stop()
        if not eleven_labs_key:
            st.error("Eleven Labs API key not found in secrets")
            st.stop()
        st.session_state.config_manager.config["openrouter_api_key"] = openrouter_key
        st.session_state.config_manager.config["eleven_labs_api_key"] = eleven_labs_key
    except Exception as e:
        st.error(f"Error accessing secrets: {e}")
        st.stop()
    
    if 'page' not in st.session_state:
        st.session_state.page = 'login'
    if 'companion' not in st.session_state:
        st.session_state.companion = None
    if 'recording_active' not in st.session_state:
        st.session_state.recording_active = False
    if 'audio_data' not in st.session_state:
        st.session_state.audio_data = None
    
    def handle_login():
        try:
            st.session_state.page = 'main'
            logger.info("Login successful, navigating to main page")
            st.experimental_rerun()
        except Exception as e:
            logger.error(f"Login error: {e}")
            st.error(f"Login failed: {e}")
    
    if st.session_state.page == 'login':
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown('<div class="login-header">', unsafe_allow_html=True)
        st.markdown('<h1 class="app-title"><span class="app-title-halo">Halo</span><span class="app-title-ai">.AI</span></h1>', unsafe_allow_html=True)
        st.markdown('<h1 class="login-title">Let\'s Get Started!</h1>', unsafe_allow_html=True)
        st.markdown('<p class="login-subtitle">Discover the latest 1000+ Voice Effects</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("Continue with Email", key="email_login"):
            handle_login()
        
        st.markdown('<div class="or-divider">or</div>', unsafe_allow_html=True)
        
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
    
    elif st.session_state.page == 'main':
        try:
            if 'audio_processor' not in st.session_state:
                st.session_state.audio_processor = ImprovedVoiceInputProcessor(
                    transcription_method=st.session_state.config_manager.get("transcription_method", "whisper")
                )
        except Exception as e:
            logger.error(f"Failed to initialize audio processor: {e}")
            st.error("Failed to initialize audio processor. Please refresh the page.")
            st.stop()
        
        try:
            if st.session_state.companion is None:
                with st.spinner("Initializing AI companion..."):
                    st.session_state.companion = EmotionalAICompanion(
                        st.session_state.config_manager.get("openrouter_api_key"),
                        st.session_state.config_manager
                    )
        except Exception as e:
            logger.error(f"Failed to initialize AI companion: {e}")
            st.error("Failed to initialize AI companion. Please check your API keys.")
            st.stop()
        
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        st.markdown('''
        <div class="ai-avatar">
            <div class="ai-avatar-inner">
                <div class="ai-avatar-face">
                    👨
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            audio_component = audio_recorder_component()
        
        if audio_component:
            logger.info(f"Audio component received, type: {type(audio_component)}, starts with data:audio/webm: {isinstance(audio_component, str) and audio_component.startswith('data:audio/webm')}")
            if isinstance(audio_component, str) and audio_component.startswith('data:audio/webm'):
                logger.info(f"Processing audio component: {audio_component[:50]}...")
                with st.spinner("Processing audio..."):
                    try:
                        audio_base64 = audio_component.split(',')[1]
                        audio_bytes = BytesIO(base64.b64decode(audio_base64))
                        audio_data = st.session_state.audio_processor.process_webm_audio(audio_bytes)
                        st.session_state.audio_data = None
                        
                        if audio_data is not None:
                            logger.info(f"Audio data shape: {audio_data.shape}")
                            transcribed_text = st.session_state.audio_processor.transcribe_audio(audio_data)
                            logger.info(f"Transcribed text: {transcribed_text}")
                            if transcribed_text:
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
                            logger.error("No audio data processed")
                            st.error("❌ No audio captured. Please try again.")
                        
                        logger.info("Audio processed, triggering rerun")
                        time.sleep(1)
                        st.experimental_rerun()
                    
                    except Exception as e:
                        logger.error(f"Audio processing error: {e}")
                        st.error(f"Processing failed: {e}")
                        st.session_state.audio_data = None
                        st.experimental_rerun()
            else:
                logger.warning("Invalid audio component received")
                st.warning("⚠️ Invalid audio data. Please try recording again.")
        
        # Fallback manual audio upload for debugging
        with col2:
            uploaded_audio = st.file_uploader("Upload audio file (WAV/WebM) for testing", type=["wav", "webm"])
            if uploaded_audio:
                logger.info(f"Uploaded audio file: {uploaded_audio.name}")
                with st.spinner("Processing uploaded audio..."):
                    try:
                        audio_data = st.session_state.audio_processor.process_webm_audio(uploaded_audio)
                        if audio_data is not None:
                            transcribed_text = st.session_state.audio_processor.transcribe_audio(audio_data)
                            if transcribed_text:
                                interaction = st.session_state.companion.process_audio(audio_data, transcribed_text)
                                if interaction:
                                    st.success(f"You said: {transcribed_text}")
                                    st.info(f"AI: {interaction['ai_response']}")
                                else:
                                    st.warning("No response generated")
                            else:
                                st.warning("⚠️ Could not transcribe uploaded audio.")
                        else:
                            st.error("❌ No audio data processed from upload.")
                        st.experimental_rerun()
                    except Exception as e:
                        logger.error(f"Uploaded audio processing error: {e}")
                        st.error(f"Processing uploaded audio failed: {e}")
        
        st.warning("⚠️ If audio recording fails, enter text to interact or upload an audio file.")
        user_text = st.text_input("Enter your message:", key="text_input")
        if user_text:
            dummy_audio = np.array([], dtype=np.float32)
            interaction = st.session_state.companion.process_audio(dummy_audio, user_text)
            if interaction:
                st.success(f"You said: {user_text}")
                st.info(f"AI: {interaction['ai_response']}")
            st.experimental_rerun()
        
        # Test TTS directly
        with col2:
            if st.button("Test TTS"):
                logger.info("Testing TTS with sample text")
                st.session_state.companion.tts_engine.speak_text("Hello, this is a test of the text-to-speech system.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        st.error("Unknown page state")
        if st.button("Go to Login"):
            st.session_state.page = 'login'
            st.experimental_rerun()

if __name__ == "__main__":
    main()