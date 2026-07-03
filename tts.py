from gtts import gTTS

def text_to_speech(text, output_path="output.mp3"):
    tts = gTTS(text=text, lang="ru")
    tts.save(output_path)
    return output_path
