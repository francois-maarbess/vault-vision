import os
import cv2
import base64
from moviepy import VideoFileClip
from groq import Groq
from dotenv import load_dotenv

# Load environment variables (API Keys)
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def encode_image_to_base64(image_path):
    """Converts an image to base64 so Llama-Vision can 'see' it."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def describe_frame_with_vision(image_path):
    """Uses Llama-3.2-Vision to analyze the video frame."""
    base64_image = encode_image_to_base64(image_path)
    
    try:
        response = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe exactly what is happening in this video frame. Read any on-screen text, identify UI elements, or describe the physical action. Keep it under 2 sentences."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            temperature=0.2, # Low temp for factual descriptions
        )
        return response.choices.message.content
    except Exception as e:
        print(f"Vision API Error: {e}")
        return "Visual data unavailable."

def process_multimodal_video(video_path, extraction_interval=30):
    """
    THE MASTER ENGINE:
    1. Extracts Audio -> Transcribes with Whisper
    2. Extracts Frames -> Analyzes with Llama-Vision
    3. Merges them into a searchable Multimodal Timeline
    """
    print(f"🚀 Initializing VaultVision Engine for: {video_path}")
    base_name = os.path.basename(video_path).split('.')
    audio_path = f"data/uploads/{base_name}.mp3"
    
    # --- 1. AUDIO PIPELINE ---
    print("🎧 Stripping audio from video...")
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, logger=None)
    
    print("📝 Transcribing audio via Groq Whisper-v3...")
    with open(audio_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=(audio_path, file.read()),
            model="whisper-large-v3",
            response_format="verbose_json",
        )
    audio_segments = transcription.segments # List of dicts with 'start', 'end', 'text'

    # --- 2. VISION PIPELINE ---
    print(f"👁️ Extracting visual context every {extraction_interval} seconds...")
    visual_segments = []
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    
    count = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret: break
        
        # Extract frame based on interval
        if count % int(fps * extraction_interval) == 0:
            timestamp = count / fps
            frame_path = f"data/uploads/{base_name}_frame_{int(timestamp)}.jpg"
            cv2.imwrite(frame_path, frame)
            
            # Send to Llama-Vision
            print(f"   -> Analyzing frame at {int(timestamp)}s...")
            description = describe_frame_with_vision(frame_path)
            
            visual_segments.append({
                "timestamp": timestamp,
                "description": description,
                "frame_path": frame_path
            })
            
            # Clean up image to save space (optional, keep if you want to display them in UI)
            # os.remove(frame_path) 
            
        count += 1
    video.release()

    # --- 3. THE MERGE (Data Alignment) ---
    print("🧬 Fusing Audio and Visual data streams...")
    # We return both so the Vector Brain can index the transcript AND the screen contents
    
    return {
        "audio_data": audio_segments,
        "visual_data": visual_segments
    }