import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Configuração da GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Configuração do MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Função para obter os landmarks faciais
def get_face_landmarks(image_path):
    # Carrega a imagem
    image = cv2.imread(image_path)

    # Inicializa o detector de landmarks faciais
    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        # Processa a imagem
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Verifica se foram encontrados landmarks
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0].landmark  # Retorna os landmarks do primeiro rosto
    return []  # Retorna uma lista vazia se nenhum rosto for encontrado

# Função para calcular a distância Euclidiana entre dois pontos
def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# Função para calcular a distância média entre dois conjuntos de landmarks
def calculate_average_distance(landmarks1, landmarks2):
    distances = [calculate_distance(p1.x, p1.y, p2.x, p2.y) for p1, p2 in zip(landmarks1, landmarks2)]
    return np.mean(distances)

# Função para desenhar landmarks na imagem
def draw_landmarks(image, landmarks):
    for landmark in landmarks:
        x = int(landmark.x * image.shape[1])
        y = int(landmark.y * image.shape[0])
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

# Caminhos das imagens
image_path1 = 'test1.jpeg'
image_path2 = 'test.jpg'

# Obtém os landmarks das duas imagens
landmarks1 = get_face_landmarks(image_path1)
landmarks2 = get_face_landmarks(image_path2)

# Verifica se foram encontrados landmarks em ambas as imagens
if landmarks1 and landmarks2:
    # Calcula a distância média entre os landmarks dos dois rostos
    average_distance = calculate_average_distance(landmarks1, landmarks2)
    print(f"Distância média entre os landmarks: {average_distance}")

    # Define um limiar de distância para determinar se são a mesma pessoa
    threshold = 0.1  # Valor de limiar; ajuste conforme necessário
    if average_distance < threshold:
        print("Os rostos são da mesma pessoa.")
    else:
        print("Os rostos não são da mesma pessoa.")

    # Exibe as imagens com os landmarks traçados
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)
    draw_landmarks(image1, landmarks1)
    draw_landmarks(image2, landmarks2)
    cv2.imshow("Imagem 1", image1)
    cv2.imshow("Imagem 2", image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Não foi possível encontrar landmarks em uma ou ambas as imagens.")
