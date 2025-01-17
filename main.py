import streamlit as st
import os
import cv2
import numpy as np
from utils.clustering import grouping_faces
from utils.detector_factory import DetectorFactory
from time import time
import csv
from PIL import Image
from utils.dirs import del_group_dir
import pandas as pd
from streamlit_option_menu import option_menu

detector = DetectorFactory.create_detector('dlib', predictor_model="media\shape_predictor_68_face_landmarks.dat", model_describer="media\dlib_face_recognition_resnet_model_v1.dat", threshold=0.588888888888888)
# detector = DetectorFactory.create_detector("retina", model_name="ArcFace", detector_backend="retinaface")

def recognized_face():
    st.title("Reconhecimento facial")
    st.write("Faça o upload de múltiplas imagens para realizar o reconhecimento facial")

    uploaded_files = st.file_uploader("Escolha as imagens", accept_multiple_files=True, type=["jpg", "png", "jpeg"])

    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "recognized_data" not in st.session_state:
        st.session_state.recognized_data = []
    if "recognition_done" not in st.session_state:
        st.session_state.recognition_done = False

    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        os.makedirs("imagens_to_recognized", exist_ok=True)
        os.makedirs("faces_recognizeds", exist_ok=True)

        for file in uploaded_files:
            with open(os.path.join("imagens_to_recognized", file.name), "wb") as f:
                f.write(file.getbuffer())

    if st.button("Realizar o reconhecimento") and st.session_state.uploaded_files:
        start_time = time()
        detector.run("imagens_to_recognized")
        execution_time = time() - start_time
        st.success(f"Reconhecimento realizado com sucesso! Tempo de execução: {execution_time:.2f} segundos")

        st.session_state.recognition_done = True

        recognized_data = []
        for root, _ , files in os.walk("faces_recognizeds"):
            for label_faces in files:
                label_faces_subheader = label_faces.split(".")[0]
                for label in os.listdir("faces"):
                    label_faces_subheader = label_faces_subheader.split("_")[0]
                    label_subheader = label.split(".")[0]
                    if label_faces_subheader.upper() in label_subheader.upper():
                        image_path_test = os.path.join("faces", label)
                        if os.path.exists(image_path_test):
                            recognized_data.append({
                                "label": label_faces_subheader,
                                "faces_recognized": os.path.join(root, label_faces),
                                "faces_to": image_path_test
                            })
                            break
        
        st.session_state.recognized_data = recognized_data

    if st.session_state.recognition_done:
        st.subheader("Tabelamento:")
        csv_data = []

        for idx, data in enumerate(st.session_state.recognized_data):
            label_faces_subheader = data["label"]
            faces_recognized = data["faces_recognized"]
            faces_to = data["faces_to"]

            st.subheader(f"{label_faces_subheader}")
            col1, col2 = st.columns([2, 2])

            writer_entry = {"recog": None, "correct": None}

            with col1:
                if os.path.exists(faces_recognized):
                    image = Image.open(faces_recognized)
                    st.image(image, caption=os.path.basename(image.filename), width=100)
                    writer_entry["recog"] = os.path.splitext(os.path.basename(image.filename))[0]

            with col2:
                if os.path.exists(faces_to):
                    image = Image.open(faces_to)
                    st.image(image, caption=os.path.basename(image.filename), width=250)
                    writer_entry["correct"] = os.path.splitext(os.path.basename(image.filename))[0]

            csv_data.append(writer_entry)

        csv_file = 'data.csv'
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["recog", "correct"])
            writer.writeheader()
            writer.writerows(csv_data)

        st.success(f"Dados salvos em {csv_file}")
        
def show_results():
    csv_file = 'data.csv'
    if os.path.exists(csv_file):
        data = []
        with open(csv_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append({
                    "recog": row["recog"],
                    "correct": row["correct"]
                })

        results = {}
        total_correct = 0
        total_incorrect = 0

        for row in data:
            recog = row["recog"]
            correct = row["correct"]
            if recog not in results:
                results[recog] = {"correct": 0, "incorrect": 0}
            if recog.split("_")[0] in correct:
                results[recog]["correct"] += 1
                total_correct += 1
            else:
                results[recog]["incorrect"] += 1
                total_incorrect += 1

        results_df = pd.DataFrame([
            {"Nome": recog, "Acertos": counts["correct"], "Erros": counts["incorrect"]}
            for recog, counts in results.items()
        ])

        total_responses = total_correct + total_incorrect
        percent_correct = (total_correct / total_responses) * 100 if total_responses > 0 else 0
        percent_incorrect = (total_incorrect / total_responses) * 100 if total_responses > 0 else 0

        summary_row = pd.DataFrame([{
            "Nome": "Total Geral",
            "Acertos": f"{percent_correct:.2f}%",
            "Erros": f"{percent_incorrect:.2f}%"
        }])

        results_df = pd.concat([results_df, summary_row], ignore_index=True)

        st.title("Resultados Gerais")
        st.subheader("Taxas de acertos e erros para o reconhecimento facial gerado.")
        st.dataframe(results_df, height=600, width=1100)
    else:
        st.warning("O arquivo `data.csv` não foi encontrado. Por favor, execute o processo de reconhecimento primeiro.")

def grouping():
    st.title("Agrupamento de rostos")
    st.info("Certifique-se de ter feito o reconhecimento facial para agrupar os rostos.")
    if st.button("Agrupar rostos"):
        sart_time = time()
        # for file in os.listdir("faces_recognizeds"):
        #     path = os.path.join("faces_recognizeds", file)                                    
        grouping_faces("faces_recognizeds")
        execution_time = time() - sart_time
        st.success(f"Agrupamento realizado com sucesso! Tempo de execução: {execution_time:.2f} segundos")
        st.write("Imagens agrupadas:")
        for i, label in enumerate(os.listdir("cluster")):
            st.subheader(f"{label} {i + 1}")
            cluster_path = os.path.join("cluster", label)
            images_in_group = os.listdir(cluster_path)

            for image_file in images_in_group:
                image_path = os.path.join(cluster_path, image_file)
                st.image(image_path, caption=image_file, width=200)

def main():
    menu = option_menu(
        menu_title="",
        options=["Reconhecimento Facial", "Visualizar Resultados", "Agrupamento de Rostos"],
        icons=["camera", "graph-up", "people"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#f9f9f9"},
            "nav-link": {"font-size": "14px", "text-align": "center", "margin": "0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#02ab21"},
        },
    )

    match menu:
        case "Reconhecimento Facial":
            recognized_face()
        case "Visualizar Resultados":
            show_results()     
        case "Agrupamento de Rostos":
            grouping()

if __name__ == "__main__":
    # detector.extract_faces()
    main()
