import os
import torch
import gradio as gr
import numpy as np
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
from pydub import AudioSegment
import tempfile
import shutil
import subprocess
import warnings
warnings.filterwarnings("ignore")

# Check if FFmpeg is properly installed
def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("✅ FFmpeg is properly installed and available in PATH")
        return True
    except FileNotFoundError:
        print("❌ FFmpeg is not found in PATH. Audio processing may fail.")
        return False

# Setup the environment and models
print("Setting up the Whisper model...")

# Check CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Download and load the whisper model (small size for balanced performance and accuracy)
model_name = "openai/whisper-small"  # You can also use "openai/whisper-tiny" for better performance on limited resources

# Initialize the model
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)
model = model.to(device)

# Create a speech recognition pipeline
pipe = pipeline(
    task="automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=8,
    device=device,
)

print("Model loaded successfully!")

def transcribe_audio(audio, language="english"):
    """Transcribe audio file using the Whisper model with enhanced error handling."""
    try:
        # Additional validation
        if audio is None:
            return "Error: No audio provided."
        
        if isinstance(audio, str) and not os.path.exists(audio):
            return f"Error: Audio file not found at path: {audio}"
            
        # Set the language for transcription
        if language == "hindi":
            forced_decoder_ids = processor.get_decoder_prompt_ids(language="hi", task="transcribe")
            pipe.model.config.forced_decoder_ids = forced_decoder_ids
        else:  # Default to English
            forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
            pipe.model.config.forced_decoder_ids = forced_decoder_ids
        
        # Process the audio with detailed error reporting
        print(f"Processing audio file: {audio}")
        result = pipe(audio, generate_kwargs={"language": "en" if language == "english" else "hi"})
        print("Transcription completed successfully")
        
        return result["text"]
    except Exception as e:
        print(f"Transcription error details: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error during transcription: {str(e)}"

def process_uploaded_file(file_path, language="english"):
    """Process an uploaded audio file for transcription with robust error handling."""
    try:
        if file_path is None:
            return "Error: No file uploaded."
            
        print(f"Processing file: {file_path}")
        
        # Create a temporary directory to work with files
        temp_dir = tempfile.mkdtemp()
        temp_audio_path = os.path.join(temp_dir, "temp_audio.wav")
        
        try:
            # Handle MP3 files by converting to WAV
            if str(file_path).lower().endswith('.mp3'):
                print(f"Converting MP3 to WAV: {file_path} -> {temp_audio_path}")
                sound = AudioSegment.from_mp3(file_path)
                sound.export(temp_audio_path, format="wav")
                file_to_process = temp_audio_path
            else:
                # For other audio formats, just use the file directly
                file_to_process = file_path
                
            # Verify the file exists before transcription
            if not os.path.exists(file_to_process):
                return f"Error: Audio file not found at {file_to_process}"
                
            print(f"File ready for transcription: {file_to_process}")
            
            # Transcribe the audio
            transcript = transcribe_audio(file_to_process, language)
            return transcript
            
        finally:
            # Always clean up temporary files
            try:
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as cleanup_error:
                print(f"Warning: Failed to clean up temp files: {cleanup_error}")
                
    except Exception as e:
        print(f"File processing error details: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error processing file: {str(e)}"

def create_gradio_interface():
    """Create and launch the Gradio interface."""
    
    # Define theme and styling
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="indigo",
    )
    
    # Create the interface
    with gr.Blocks(theme=theme, title="Whispr.ai") as app:
        gr.Markdown(
            """
            # 🎙️ Whispr.ai
            
            Convert speech to text using Whisper AI model. Supports English and Hindi languages.
            
            - Record audio directly or upload MP3 files
            - Select your preferred language
            - Get accurate transcriptions instantly
            """
        )
        
        with gr.Row():
            language_radio = gr.Radio(
                ["english", "hindi"], 
                label="Select Language", 
                value="english",
                info="Choose the language of your audio"
            )
        
        with gr.Tabs():
            with gr.TabItem("🎤 Record Audio"):
                with gr.Row():
                    audio_input = gr.Audio(
                        sources=["microphone"], 
                        type="filepath",
                        label="Record your voice"
                    )
                    
                with gr.Row():
                    record_button = gr.Button("Transcribe Recording", variant="primary")
                
                with gr.Row():
                    output_text = gr.Textbox(
                        lines=5, 
                        label="Transcription Result",
                        placeholder="Your transcription will appear here..."
                    )
                
                record_button.click(
                    fn=process_uploaded_file,
                    inputs=[audio_input, language_radio],
                    outputs=output_text
                )
            
            with gr.TabItem("📁 Upload MP3"):
                with gr.Row():
                    file_input = gr.File(
                        label="Upload MP3 File",
                        file_types=[".mp3", ".wav"]
                    )
                
                with gr.Row():
                    upload_button = gr.Button("Transcribe File", variant="primary")
                
                with gr.Row():
                    upload_output = gr.Textbox(
                        lines=5, 
                        label="Transcription Result",
                        placeholder="Your transcription will appear here..."
                    )
                
                upload_button.click(
                    fn=process_uploaded_file,
                    inputs=[file_input, language_radio],
                    outputs=upload_output
                )
        
        gr.Markdown(
            """
            ### Instructions
            1. Select your preferred language (English or Hindi)
            2. Either record audio using the microphone or upload an MP3 file
            3. Click the transcribe button to convert speech to text
            4. View the transcription result below
            
            For best results, ensure clear audio with minimal background noise.
            """
        )
    
    return app

# Launch the app
if __name__ == "__main__":
    # Final verification that FFmpeg is available
    check_ffmpeg()
    
    print("Launching the Whispr.ai application...")
    app = create_gradio_interface()
    app.launch()