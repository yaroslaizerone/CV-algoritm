import os

import cv2

# Путь к видеофайлам и каскадам
video_folder = "input_video"
cascade_folder = "haarcascades"

# Список каскадов (файл: описание)
haar_cascades = {
    "haarcascade_eye.xml": "Глаза",
    "haarcascade_eye_tree_eyeglasses.xml": "Глаза в очках",
    "haarcascade_frontalcatface.xml": "Фронтальная кошачья морда",
    "haarcascade_frontalcatface_extended.xml": "Фронтальная кошачья морда",
    "haarcascade_frontalface_alt.xml": "Фронтальное лицо 1",
    "haarcascade_frontalface_alt2.xml": "Фронтальное лицо 2",
    "haarcascade_frontalface_alt_tree.xml": "Фронтальное лицо каскад",
    "haarcascade_frontalface_default.xml": "Фронтальное лицо по умолчанию",
    "haarcascade_fullbody.xml": "Полное тело",
    "haarcascade_lefteye_2splits.xml": "Левый глаз",
    "haarcascade_licence_plate_rus_16stages.xml": "Российские номерные знаки",
    "haarcascade_lowerbody.xml": "Нижняя часть тела",
    "haarcascade_profileface.xml": "Профиль лица",
    "haarcascade_righteye_2splits.xml": "Правый глаз",
    "haarcascade_russian_plate_number.xml": "Номерные знаки",
    "haarcascade_smile.xml": "Улыбка",
    "haarcascade_upperbody.xml": "Верхняя часть тела"
}

# Получаем список всех видеофайлов
video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]

for video_file in video_files:
    print(f"\n🔍 Обработка видео: {video_file}")
    cap = cv2.VideoCapture(os.path.join(video_folder, video_file))

    if not cap.isOpened():
        print(f"❌ Не удалось открыть видео: {video_file}")
        continue

    for cascade_file, description in haar_cascades.items():
        cascade_path = os.path.join(cascade_folder, cascade_file)
        if not os.path.exists(cascade_path):
            print(f"⚠️ Каскад не найден: {cascade_path}")
            continue

        print(f"  ▶️ Применение каскада: {description} ({cascade_file})")
        detector = cv2.CascadeClassifier(cascade_path)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # возвращаемся к началу видео

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            objects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in objects:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            label = f"{description} ({cascade_file})"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            cv2.imshow("Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    cap.release()

cv2.destroyAllWindows()
