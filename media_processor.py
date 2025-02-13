
import os
import tempfile
import logging
import ffmpeg
from typing import Optional, Dict, Any
from langchain.schema import Document

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MediaProcessor:
    def __init__(self):
        """Initialize media processor"""
        logger.info("Initializing media processor...")

    def extract_audio_from_video(self, video_path: str) -> Optional[str]:
        """Extract audio from video file"""
        try:
            temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            logger.info(f"Extracting audio from video: {video_path}")
            
            ffmpeg.input(video_path).output(
                temp_audio.name,
                acodec='pcm_s16le',
                ac=1,
                ar='16k'
            ).overwrite_output().run(quiet=True)
            
            logger.info(f"Audio extracted successfully to {temp_audio.name}")
            return temp_audio.name
        except Exception as e:
            logger.error(f"Error extracting audio from video: {e}")
            return None

    def get_media_metadata(self, file_path: str) -> Dict[str, Any]:
        """Get metadata for media file"""
        try:
            probe = ffmpeg.probe(file_path)
            metadata = {
                'format': probe['format']['format_name'],
                'duration': float(probe['format']['duration']),
                'size': int(probe['format']['size']),
                'streams': []
            }
            
            for stream in probe['streams']:
                stream_type = stream['codec_type']
                stream_info = {
                    'type': stream_type,
                    'codec': stream['codec_name']
                }
                if stream_type == 'video':
                    stream_info.update({
                        'width': int(stream['width']),
                        'height': int(stream['height']),
                        'fps': eval(stream.get('r_frame_rate', '0/1'))
                    })
                elif stream_type == 'audio':
                    stream_info.update({
                        'channels': int(stream.get('channels', 1)),
                        'sample_rate': int(stream.get('sample_rate', 0))
                    })
                metadata['streams'].append(stream_info)
            
            return metadata
        except Exception as e:
            logger.error(f"Error getting media metadata: {e}")
            return {}

    def process_media_file(self, file_path: str) -> Optional[Document]:
        """Process media file and return document with metadata"""
        try:
            # Get file type
            ext = os.path.splitext(file_path)[1].lower()
            metadata = self.get_media_metadata(file_path)
            
            if ext in ['.mp4', '.avi', '.mov', '.mkv']:
                # For video files, extract audio and create document
                audio_path = self.extract_audio_from_video(file_path)
                if audio_path:
                    try:
                        # Create document with metadata
                        doc = Document(
                            page_content=f"Video file processed: {file_path}",
                            metadata={
                                'source': file_path,
                                'modality': 'video',
                                'extracted_audio': audio_path,
                                **metadata
                            }
                        )
                        return doc
                    finally:
                        # Clean up temporary audio file
                        try:
                            os.unlink(audio_path)
                        except:
                            pass
            
            elif ext in ['.mp3', '.wav', '.m4a', '.ogg']:
                # For audio files, create document with metadata
                doc = Document(
                    page_content=f"Audio file processed: {file_path}",
                    metadata={
                        'source': file_path,
                        'modality': 'audio',
                        **metadata
                    }
                )
                return doc
            
            logger.info(f"Successfully processed media file: {file_path}")
            return None
            
        except Exception as e:
            logger.error(f"Error processing media file {file_path}: {e}")
            return None
