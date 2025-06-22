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

# Configuration management with secure API key handling
class ConfigManager:
    """Manages persistent settings for the application with secure API key handling"""
    def __init__(self, config_file="config.json"):
        self.config_file = config_file
        self.config = {
            "openrouter_api_key": self._get_api_key("openrouter_api_key"),
            "eleven_labs_api_key": self._get_api_key("eleven_labs_api_key"),
            "voice_id": "21m00Tcm4TlvDq8ikWAM",
            "silence_threshold": 0.01,
            "max_duration": 10,
            "transcription_method": "whisper"
        }
        self.load_config()
    
    def _get_api_key(self, key_name):
        """Get API key from Streamlit secrets, environment variables, or session state"""
        try:
            # Try session state first (for user-input keys)
            if f"user_{key_name}" in st.session_state:
                return st.session_state[f"user_{key_name}"]
            
            # Try Streamlit secrets
            if hasattr(st, 'secrets') and key_name in st.secrets:
                return st.secrets[key_name]
            
            # Try environment variables
            env_key = key_name.upper()
            if env_key in os.environ:
                return os.environ[env_key]
            
            # Default placeholder
            return "enter-api"
        except Exception as e:
            logger.warning(f"Could not retrieve {key_name}: {e}")
            return "enter-api"
    
    def load_config(self):
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    saved_config = json.load(f)
                    # Only load non-sensitive settings from file
                    non_sensitive_keys = ["voice_id", "silence_threshold", "max_duration", "transcription_method"]
                    for key in non_sensitive_keys:
                        if key in saved_config:
                            self.config[key] = saved_config[key]
                logger.info(f"Loaded non-sensitive config from {self.config_file}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
    
    def save_config(self):
        try:
            # Only save non-sensitive settings to file
            non_sensitive_config = {
                "voice_id": self.config["voice_id"],
                "silence_threshold": self.config["silence_threshold"],
                "max_duration": self.config["max_duration"],
                "transcription_method": self.config["transcription_method"]
            }
            with open(self.config_file, 'w') as f:
                json.dump(non_sensitive_config, f, indent=2)
            logger.info(f"Saved non-sensitive config to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def get(self, key, default=None):
        # Refresh API keys from current sources
        if key in ["openrouter_api_key", "eleven_labs_api_key"]:
            self.config[key] = self._get_api_key(key)
        return self.config.get(key, default)
    
    def set(self, key, value):
        self.config[key] = value
        if key in ["openrouter_api_key", "eleven_labs_api_key"]:
            # Store in session state for API keys
            st.session_state[f"user_{key}"] = value
        else:
            # Save non-sensitive settings to file
            self.save_config()

# API Key validation
def validate_api_keys():
    """Check if required API keys are properly configured"""
    config_manager = st.session_state.get('config_manager')
    if not config_manager:
        return False, "Configuration not initialized"
    
    openrouter_key = config_manager.get("openrouter_api_key", "")
    if not openrouter_key or openrouter_key == "enter-api":
        return False, "OpenRouter API key is required"
    
    return True, "API keys validated"

# Configuration page for API keys
def show_config_page():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    st.markdown('<h1 class="login-title">🔧 Configuration</h1>', unsafe_allow_html=True)
    st.markdown('<p class="login-subtitle">Enter your API keys to use Halo.AI</p>', unsafe_allow_html=True)
    
    with st.form("api_config"):
        st.markdown("### Required")
        openrouter_key = st.text_input(
            "OpenRouter API Key", 
            type="password",
            help="Get your API key from https://openrouter.ai/",
            placeholder="sk-or-..."
        )
        
        st.markdown("### Optional")
        eleven_labs_key = st.text_input(
            "Eleven Labs API Key", 
            type="password",
            help="For high-quality voice synthesis. Get from https://elevenlabs.io/",
            placeholder="Optional - will use robotic voice if not provided"
        )
        
        voice_id = st.text_input(
            "Eleven Labs Voice ID",
            value="21m00Tcm4TlvDq8ikWAM",
            help="Voice ID from Eleven Labs (default: Rachel)"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.form_submit_button("💾 Save & Continue", use_container_width=True):
                if openrouter_key and openrouter_key != "enter-api":
                    st.session_state.config_manager.set("openrouter_api_key", openrouter_key)
                    st.session_state.config_manager.set("eleven_labs_api_key", eleven_labs_key)
                    st.session_state.config_manager.set("voice_id", voice_id)
                    st.success("✅ Configuration saved!")
                    st.session_state.page = 'main'
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ OpenRouter API key is required")
        
        with col2:
            if st.form_submit_button("⬅️ Back to Login", use_container_width=True):
                st.session_state.page = 'login'
                st.rerun()
    
    # Info section
    with st.expander("ℹ️ How to get API keys"):
        st.markdown("""
        **OpenRouter API Key (Required):**
        1. Go to [openrouter.ai](https://openrouter.ai/)
        2. Sign up for an account
        3. Navigate to API Keys section
        4. Create a new API key
        
        **Eleven Labs API Key (Optional):**
        1. Go to [elevenlabs.io](https://elevenlabs.io/)
        2. Sign up for an account
        3. Go to your Profile → API Keys
        4. Copy your API key
        
        **Note:** Without Eleven Labs, the app will use a robotic voice for responses.
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Updated Voice Input Processor with file upload fallback
class ImprovedVoiceInputProcessor:
    """Handles audio recording and transcription with file upload fallback"""
    def __init__(self, sample_rate=16000, channels=1, transcription_method="whisper"):
        self.sample_rate = sample_rate
        self.channels = channels
        self.transcription_method = transcription_method
        if transcription_method == "whisper":
            try:
                self.whisper_model = whisper.load_model("tiny")
                logger.info("Whisper model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                st.error(f"❌ Failed to load Whisper model: {e}")
                self.whisper_model = None
        self._test_audio_setup()
    
    def _test_audio_setup(self):
        """Test if audio recording is available"""
        try:
            devices = sd.query_devices()
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            if not input_devices:
                logger.warning("No input devices found - using file upload mode")
                st.session_state.audio_mode = 'upload'
                return False
            logger.info(f"Found {len(input_devices)} input device(s)")
            st.session_state.audio_mode = 'record'
            return True
        except Exception as e:
            logger.warning(f"Audio setup not available: {e} - using file upload mode")
            st.session_state.audio_mode = 'upload'
            return False
    
    def process_uploaded_audio(self, uploaded_file):
        """Process uploaded audio file"""
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Load audio with librosa
            audio_data, sr = librosa.load(tmp_path, sr=self.sample_rate)
            
            # Clean up
            os.unlink(tmp_path)
            
            logger.info(f"Processed uploaded audio: {len(audio_data)/sr:.2f} seconds")
            return audio_data
        except Exception as e:
            logger.error(f"Error processing uploaded audio: {e}")
            return None
    
    # ... (rest of the existing methods remain the same)
    def transcribe_audio(self, audio_data: np.ndarray) -> Optional[str]:
        if audio_data is None or len(audio_data) == 0:
            return None
        try:
            if isinstance(audio_data, np.ndarray):
                # Convert numpy array to temporary wav file
                wav_file = self.save_audio_to_wav(audio_data)
            else:
                # Already a file path
                wav_file = audio_data
            
            if not wav_file:
                return None
            
            if self.transcription_method == "whisper" and self.whisper_model:
                result = self.whisper_model.transcribe(wav_file, language="en")
                if isinstance(audio_data, np.ndarray):
                    os.remove(wav_file)
                return result["text"]
            else:
                logger.warning("No valid transcription method available")
                if isinstance(audio_data, np.ndarray):
                    os.remove(wav_file)
                return ""
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            if isinstance(audio_data, np.ndarray) and os.path.exists(wav_file):
                os.remove(wav_file)
            return None
    
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

# ... (Keep all other classes: EmotionDetector, ResponseGenerator, TextToSpeechEngine, EmotionalAICompanion)
# [The rest of the classes remain exactly the same as in your original code]

# Updated CSS with config page styling
def load_css():
    st.markdown("""
    <style>
    /* ... (keep all existing CSS) ... */
    
    /* Config page specific styling */
    .config-container {
        background: #1a1a1a;
        min-height: 100vh;
        color: white;
        padding: 2rem;
    }
    
    .stTextInput > div > div > input {
        background: #2a2a2a !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
        color: white !important;
        padding: 0.75rem !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #555 !important;
        box-shadow: 0 0 0 1px #555 !important;
    }
    
    .stForm {
        background: #2a2a2a;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #333;
    }
    
    .stExpander > details > summary {
        background: #2a2a2a !important;
        color: white !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
    }
    
    .stExpander > details > div {
        background: #2a2a2a !important;
        border: 1px solid #333 !important;
        border-top: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Updated main function
def main():
    st.set_page_config(page_title="Halo.ai", page_icon="🤖", layout="wide")
    
    # Load custom CSS
    load_css()
    
    # Initialize ConfigManager
    if 'config_manager' not in st.session_state:
        st.session_state.config_manager = ConfigManager()
    
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'login'
    if 'companion' not in st.session_state:
        st.session_state.companion = None
    if 'audio_mode' not in st.session_state:
        st.session_state.audio_mode = 'upload'  # Default to upload mode
    
    # Helper function to handle login
    def handle_login():
        try:
            # Check if API keys are configured
            is_valid, message = validate_api_keys()
            if is_valid:
                st.session_state.page = 'main'
                logger.info("Login successful with valid API keys, navigating to main page")
            else:
                st.session_state.page = 'config'
                logger.info("Login successful but API keys needed, navigating to config page")
            st.rerun()
        except Exception as e:
            logger.error(f"Login error: {e}")
            st.error(f"Login failed: {e}")
    
    # Login Page
    if st.session_state.page == 'login':
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown('<div class="login-header">', unsafe_allow_html=True)
        st.markdown('<h1 class="app-title"><span class="app-title-halo">Halo</span><span class="app-title-ai">.AI</span></h1>', unsafe_allow_html=True)
        st.markdown('<h1 class="login-title">Let\'s Get Started!</h1>', unsafe_allow_html=True)
        st.markdown('<p class="login-subtitle">Your AI Emotional Companion</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Login buttons
        if st.button("🚀 Start Using Halo.AI", key="start_button"):
            handle_login()
        
        st.markdown('<div class="or-divider">or</div>', unsafe_allow_html=True)
        
        if st.button("⚙️ Configure API Keys", key="config_button"):
            st.session_state.page = 'config'
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Configuration Page
    elif st.session_state.page == 'config':
        show_config_page()
    
    # Main Interaction Page
    elif st.session_state.page == 'main':
        try:
            # Validate API keys before proceeding
            is_valid, message = validate_api_keys()
            if not is_valid:
                st.error(f"❌ {message}")
                if st.button("⚙️ Configure API Keys"):
                    st.session_state.page = 'config'
                    st.rerun()
                return
            
            # Initialize components
            if 'audio_processor' not in st.session_state:
                st.session_state.audio_processor = ImprovedVoiceInputProcessor(
                    transcription_method=st.session_state.config_manager.get("transcription_method", "whisper")
                )
            
            if st.session_state.companion is None:
                with st.spinner("Initializing AI companion..."):
                    openrouter_api_key = st.session_state.config_manager.get("openrouter_api_key")
                    st.session_state.companion = EmotionalAICompanion(
                        openrouter_api_key,
                        st.session_state.config_manager
                    )
            
            # Render the main page
            st.markdown('<div class="main-container">', unsafe_allow_html=True)
            
            # AI Avatar
            st.markdown('''
            <div class="ai-avatar">
                <div class="ai-avatar-inner">
                    <div class="ai-avatar-face">
                        🤖
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            # Audio input method selection
            audio_mode = st.radio(
                "Choose input method:",
                ["Upload Audio File", "Record Audio (if supported)"],
                key="audio_input_method",
                horizontal=True
            )
            
            if "Upload" in audio_mode:
                # File upload mode
                uploaded_file = st.file_uploader(
                    "Upload an audio file",
                    type=['wav', 'mp3', 'ogg', 'm4a'],
                    help="Record a voice message and upload it here"
                )
                
                if uploaded_file is not None:
                    with st.spinner("Processing your audio..."):
                        try:
                            audio_data = st.session_state.audio_processor.process_uploaded_audio(uploaded_file)
                            if audio_data is not None:
                                transcribed_text = st.session_state.audio_processor.transcribe_audio(audio_data)
                                if transcribed_text:
                                    logger.info(f"Transcribed text: {transcribed_text}")
                                    interaction = st.session_state.companion.process_audio(audio_data, transcribed_text)
                                    if interaction:
                                        st.success(f"You said: {transcribed_text}")
                                        st.info(f"AI: {interaction['ai_response']}")
                                    else:
                                        st.warning("No response generated")
                                else:
                                    st.warning("⚠️ Could not transcribe audio. Please try again.")
                            else:
                                st.error("❌ Could not process audio file.")
                        except Exception as e:
                            logger.error(f"Processing error: {e}")
                            st.error(f"Processing failed: {e}")
            
            else:
                # Recording mode (may not work on Streamlit Cloud)
                st.warning("⚠️ Live recording may not work on Streamlit Cloud. Use file upload instead.")
                
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    if st.button("🎙️ Try Recording", key="record_button"):
                        st.info("Recording functionality requires local deployment or special hosting.")
            
            # Settings link
            if st.button("⚙️ Settings"):
                st.session_state.page = 'config'
                st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            logger.error(f"Main page error: {e}")
            st.error(f"Page loading failed: {e}")
            if st.button("⚙️ Check Configuration"):
                st.session_state.page = 'config'
                st.rerun()

if __name__ == "__main__":
    main()