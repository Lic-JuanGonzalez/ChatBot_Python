import PySimpleGUI as sg
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Cargar preguntas y respuestas desde el archivo JSON
with open('data.json', encoding='utf-8') as f:
    data = json.load(f)

# Extraer preguntas y respuestas del conjunto de datos
preguntas  = [item["pregunta"] for item in data["data"]]
respuestas = [item["respuesta"] for item in data["data"]]

# Crear un modelo de clasificación
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(preguntas, respuestas)

# Función para predecir la respuesta
def predecir_respuesta(pregunta):
    return model.predict([pregunta])[0]

# Función para mostrar el menú
def mostrar_menu():
    menu_text = "Menú de opciones:\n"
    for i, pregunta in enumerate(preguntas, start=1):
        menu_text += f"{i}. {pregunta}\n"
    menu_text += "0. Hacer una pregunta libre\n"
    menu_text += "Elije una opción: \n"
    return menu_text

def mostrar_respuesta(respuesta):
    window['respuesta_area'].print("Usuario: " + window['pregunta'].get())
    window['respuesta_area'].print("Bot: " + respuesta)
    window['respuesta_area'].print("")  
    window['respuesta_area'].print(mostrar_menu())  
    window['pregunta'].update('') 

# Crear la interfaz gráfica
layout = [
    [sg.Text('Chatbot Universidad Aconcagua')],
    [sg.Multiline(mostrar_menu(), key='respuesta_area', size=(120, 30))],
    [sg.Text("Pregunta: "), sg.InputText(key='pregunta')],
    [sg.Button("Enviar Pregunta"), sg.Button("Salir")]
]

window = sg.Window('Chatbot', layout)

# Función para iniciar el chat
def start_chat():
    event, values = window.read()

    # window['respuesta_area'].print("¡Hola! Soy tu asistente virtual.")
    while True:
        event, values = window.read()
        
        if event in (sg.WIN_CLOSED, 'Salir'):
            break
        elif event == 'Enviar Pregunta':
            user_choice = values['pregunta']
        if user_choice.isdigit() and int(user_choice) == 0:
            window['respuesta_area'].print("Bot: Hazme una pregunta")
            window['pregunta'].update('')
            # window['respuesta_area'].print(mostrar_menu())
            while True:
                event, values = window.read()
                if event in (sg.WIN_CLOSED, 'Salir'):
                    break
                elif event == 'Enviar Pregunta':
                    user_choice = values['pregunta']
                    if user_choice:
                        respuesta_predicha = predecir_respuesta(user_choice)
                        mostrar_respuesta(respuesta_predicha)
                        break
                window['respuesta_area'].print("Esperando a que ingreses una pregunta...")

        elif user_choice.isdigit() and 0 < int(user_choice) <= len(preguntas):
            respuesta_predicha = respuestas[int(user_choice) - 1]
            mostrar_respuesta(respuesta_predicha)
        else:
            respuesta_predicha = predecir_respuesta(user_choice)
            mostrar_respuesta(respuesta_predicha)

# Iniciar el chat
start_chat()
