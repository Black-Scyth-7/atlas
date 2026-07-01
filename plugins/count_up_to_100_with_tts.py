"""Atlas custom tool 'count_up_to_100_with_tts' — created via voice on 2026-06-30."""

TOOL = {
    "name": 'count_up_to_100_with_tts',
    "description": 'Counts up to 100 and speaks each number using text-to-speech.',
    "arguments": "{'start': 1, 'end': 100}",
}


import sys
import subprocess

def run(args):
    """
    Counts up to a specified end number from a start number and speaks each number
    using text-to-speech, leveraging OS-specific commands via subprocess.

    Args:
        args (dict): A dictionary containing 'start' and 'end' keys.
                     Example: {'start': 1, 'end': 100}

    Returns:
        str: A human-readable string indicating the outcome of the operation.
             Returns an error message if inputs are invalid or TTS fails.
    """
    try:
        start = int(args.get('start', 1))
        end = int(args.get('end', 100))
    except (ValueError, TypeError):
        return "Error: 'start' and 'end' must be valid integers."

    if start > end:
        return "Error: 'start' cannot be greater than 'end'."

    platform = sys.platform
    tts_base_command = []
    
    # Determine the base TTS command based on the operating system
    if platform.startswith('darwin'):  # macOS
        tts_base_command = ["say"]
    elif platform.startswith('linux'): # Linux
        # 'espeak' is a widely available TTS engine on Linux distributions.
        # Ensure 'espeak' is installed (e.g., sudo apt-get install espeak).
        tts_base_command = ["espeak"]
    elif platform.startswith('win'):   # Windows
        # Use PowerShell to invoke the .NET Speech Synthesis API.
        # This provides native TTS capabilities without third-party installs.
        # The format string '{}' will be replaced with the number to speak.
        tts_base_command = ["powershell", "-Command", "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{}')"]
    else:
        return f"Error: Text-to-speech is not supported on your operating system ({platform})."

    # Iterate through the range and speak each number
    for i in range(start, end + 1):
        try:
            number_to_speak = str(i)
            full_command = []

            if platform.startswith('win'):
                # For Windows, format the PowerShell command string with the current number
                formatted_command = tts_base_command[2].format(number_to_speak)
                full_command = [tts_base_command[0], tts_base_command[1], formatted_command]
            else:
                # For macOS/Linux, simply append the number as an argument
                full_command = tts_base_command + [number_to_speak]

            # Execute the TTS command silently (don't print subprocess output)
            subprocess.run(full_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError:
            # Provide specific help for common TTS commands
            cmd_name = tts_base_command[0] if not platform.startswith('win') else "PowerShell"
            return f"Error: '{cmd_name}' command not found. Please ensure it is installed and in your system's PATH."
        except subprocess.CalledProcessError as e:
            # Capture and decode stderr for more informative error messages
            error_output = e.stderr.decode(sys.getfilesystemencoding(), errors='ignore').strip()
            return f"Error executing TTS command for number {i}: {e}. Details: {error_output}"
        except Exception as e:
            return f"An unexpected error occurred while speaking number {i}: {e}"

    return f"Successfully counted and spoke numbers from {start} to {end}."
