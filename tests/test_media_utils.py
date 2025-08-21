"""Tests for media utilities."""

import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock

from hunyuanvideo_foley.utils.media_utils import merge_audio_video, MediaProcessingError


class TestMergeAudioVideo:
    """Test cases for merge_audio_video function."""
    
    def test_invalid_audio_path(self):
        """Test with non-existent audio file."""
        with pytest.raises(MediaProcessingError, match="Audio file not found"):
            merge_audio_video("nonexistent.wav", "video.mp4", "output.mp4")
    
    def test_invalid_video_path(self):
        """Test with non-existent video file."""
        with tempfile.NamedTemporaryFile(suffix='.wav') as audio_file:
            with pytest.raises(MediaProcessingError, match="Video file not found"):
                merge_audio_video(audio_file.name, "nonexistent.mp4", "output.mp4")
    
    @patch('subprocess.Popen')
    def test_successful_merge(self, mock_popen):
        """Test successful merge operation."""
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.wav') as audio_file, \
             tempfile.NamedTemporaryFile(suffix='.mp4') as video_file, \
             tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as output_file:
            
            # Mock successful subprocess
            mock_process = MagicMock()
            mock_process.returncode = 0
            mock_process.communicate.return_value = ("", "")
            mock_popen.return_value = mock_process
            
            result = merge_audio_video(
                audio_file.name, 
                video_file.name, 
                output_file.name
            )
            
            assert result == output_file.name
            mock_popen.assert_called_once()
            
            # Cleanup
            os.unlink(output_file.name)
    
    @patch('subprocess.Popen')
    def test_ffmpeg_failure(self, mock_popen):
        """Test ffmpeg failure handling."""
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.wav') as audio_file, \
             tempfile.NamedTemporaryFile(suffix='.mp4') as video_file:
            
            # Mock failed subprocess
            mock_process = MagicMock()
            mock_process.returncode = 1
            mock_process.communicate.return_value = ("", "FFmpeg error")
            mock_popen.return_value = mock_process
            
            with pytest.raises(MediaProcessingError, match="FFmpeg failed"):
                merge_audio_video(
                    audio_file.name, 
                    video_file.name, 
                    "output.mp4"
                )
    
    @patch('subprocess.Popen', side_effect=FileNotFoundError)
    def test_ffmpeg_not_found(self, mock_popen):
        """Test ffmpeg not found error."""
        with tempfile.NamedTemporaryFile(suffix='.wav') as audio_file, \
             tempfile.NamedTemporaryFile(suffix='.mp4') as video_file:
            
            with pytest.raises(FileNotFoundError, match="ffmpeg not found"):
                merge_audio_video(
                    audio_file.name, 
                    video_file.name, 
                    "output.mp4"
                )