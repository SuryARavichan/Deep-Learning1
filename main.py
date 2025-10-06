from fastapi import FastAPI

from EmotionClassification import classify_emotions

app = FastAPI()

@app.post('/emotion-detection')
def emotion_classification(text_body: str):
    # Находим словарь с вероятностями эмоций
    emotions = classify_emotions(text_body)
    if emotions:
        return classify_emotions(text_body)
    # Если не удалось вычислить эмоции - возвращаем ошибку
    else:
        return {
            'Error': 1,
            'description': 'Error during classification'
        }
