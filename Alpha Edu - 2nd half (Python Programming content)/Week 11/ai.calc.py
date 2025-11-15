import cv2
from cvzone.HandTrackingModule import HandDetector

# Инициализация детектора рук
detector = HandDetector(detectionCon=0.8, maxHands=2)

# !FRAME - КАДР

# Настройка захвата видео с веб-камеры
web_camera = cv2.VideoCapture(0)  # Открытие веб-камеры

# Устанавливаем размер окна камеры (увеличиваем)
web_camera.set(3, 1280)  # Ширина окна
web_camera.set(4, 720)   # Высота окна

# Переменные для хранения чисел и операции
num1 = 0
num2 = 0
operation = "+"  #! Начальная операция - сложение
result = 0  # Для хранения результата

# Функция для отображения знаков операций
def draw_operations(frame, operation):
    """
    Функция для отображения знаков операций на экране.
    Отображаются знаки для операций +, -, *, /
    """
    if operation == "+":
        cv2.putText(frame, "+", (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 10)
    elif operation == "-":
        cv2.putText(frame, "-", (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10)
    elif operation == "*":
        cv2.putText(frame, "*", (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 10)
    elif operation == "/":
        cv2.putText(frame, "/", (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 10)

while True:
    # Чтение кадра с веб-камеры
    is_camera_ok, frame = web_camera.read()

    if not is_camera_ok:
        print("CAMERA IS NOT WORKING")

    frame = cv2.flip(frame, 1)  # Отражаем кадр по горизонтали

    # Поиск рук в кадре
    hands, frame = detector.findHands(frame)

    # Сброс количества пальцев для каждой руки
    total_fingers_left = 0
    total_fingers_right = 0

    # Если руки обнаружены
    if hands:
        # Обработка каждой руки
        for i , hand in enumerate(hands): #i == 0 or i == 1
            fingers = detector.fingersUp(hand) 
            total_fingers = sum(fingers)  # Подсчитываем количество поднятых пальцев

            # Разделяем по рукам
            if i == 0:  # Левая рука
                total_fingers_left = total_fingers
            elif i == 1:  # Правая рука
                total_fingers_right = total_fingers

        # Отображаем информацию о выбранной операции
        if operation:
            cv2.putText(frame, f"Current Operation: {operation}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Вычисляем результат в зависимости от выбранной операции
        if operation == "+":
            result = total_fingers_left + total_fingers_right
        elif operation == "-":
            result = total_fingers_left - total_fingers_right
        elif operation == "*":
            result = total_fingers_left * total_fingers_right
        elif operation == "/":
            if total_fingers_right != 0:  # Защита от деления на 0
                result = total_fingers_left / total_fingers_right
            else:
                result = "Error (div by 0)"

        # Отображаем операцию и результат
        cv2.putText(frame, f"Left Hand: {total_fingers_left}", (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Right Hand: {total_fingers_right}", (10, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Result: {result}", (10, 250), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Отображаем выбранную операцию на экране
        # draw_operations(frame, operation)

    # Отображаем окно с видео
    cv2.imshow("AI CALCULATOR", frame)

    # Выход из программы по нажатию клавиши 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  
        break
    elif key == ord('+'):
        operation = "+"  # Установить операцию на сложение
    elif key == ord('-'):
        operation = "-"  # Установить операцию на вычитание
    elif key == ord('*'):
        operation = "*"  # Установить операцию на умножение
    elif key == ord('/'):
        operation = "/"  # Установить операцию на деление

# Освобождаем захват видео и закрываем окно
web_camera.release()
cv2.destroyAllWindows()