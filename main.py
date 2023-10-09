import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Cargar preguntas y respuestas desde el archivo JSON
with open('data.json', encoding='utf-8') as f:
    data = json.load(f)

# Extraer preguntas y respuestas del conjunto de datos
preguntas = [item["pregunta"] for item in data["data"]]
respuestas = [item["respuesta"] for item in data["data"]]

# Crear un modelo de clasificación
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(preguntas, respuestas)

# Función para predecir la respuesta
def predecir_respuesta(pregunta):
    return model.predict([pregunta])[0]

# Función para mostrar el menú
def mostrar_menu():
    print("Menú de opciones:")
    for i, pregunta in enumerate(preguntas, start=1):
        print(f"{i}. {pregunta}")
    print("0. Hacer una pregunta libre")
    print("salir para salir")

# Función para iniciar el chat
def start_chat():
    print("¡Hola! Soy tu asistente virtual.")
    while True:
        mostrar_menu()
        user_choice = input("Elige una opción: ")
        if user_choice.lower() == "salir":
            print("¡Hasta luego!")
            break
        elif user_choice.isdigit() and 0 <= int(user_choice) <= len(preguntas):
            if int(user_choice) == 0:
                user_input = input("Hazme una pregunta: ")
                respuesta_predicha = predecir_respuesta(user_input)
                print(f"Bot: {respuesta_predicha}")
            else:
                respuesta_predicha = respuestas[int(user_choice) - 1]
                print(f"Bot: {respuesta_predicha}")
        else:
            print("Opción no válida. Inténtalo de nuevo.")

# Iniciar el chat
start_chat()
