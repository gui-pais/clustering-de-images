import streamlit as st
import os
from time import time
from PIL import Image
from streamlit_option_menu import option_menu
from utils.pipeline import pipeline


def recognized():
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

        for file in uploaded_files:
            with open(os.path.join("uploads", file.name), "wb") as f:
                f.write(file.getbuffer())

    if st.button("Realizar o reconhecimento") and st.session_state.uploaded_files:
        start_time = time()
        pipeline()
        execution_time = time() - start_time
        st.success(f"Reconhecimento realizado com sucesso! Tempo de execução: {execution_time:.2f} segundos")

        st.session_state.recognition_done = True

        recognized_data = []
        for root, _ , files in os.walk("output"):
            for label_faces in files:
                output = os.path.join(root, label_faces)
                recognized_data.append({"output": output})
        
        st.session_state.recognized_data = recognized_data

    if st.session_state.recognition_done:

        for data in st.session_state.recognized_data:
            output = data["output"]
            if os.path.exists(output):
                image = Image.open(output)
                st.image(image, caption=os.path.basename(image.filename), width=1000)
                    
                    
def main():
    menu = option_menu(
        menu_title="",
        options=["Reconhecimento Facial"],
        icons=["camera"],
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
            recognized()    

if __name__ == "__main__":
    main()
