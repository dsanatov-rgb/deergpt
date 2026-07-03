from gtts import gTTS

text = "Это тест преобразования текста в речь с помощью gTTS."
try:
    tts = gTTS(text, lang='ru')
    tts.save("output.mp3")
    print("Файл создан успешно!")
except Exception as e:
    print("Произошла ошибка:", e)
