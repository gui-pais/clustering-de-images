import streamlit as st
import os
import random 
from clustering import group_faces

if __name__ == "__main__":
    st.title("Identificador de Rostos")
    st.write("Faça o upload de múltiplas imagens para agrupá-las com base em sua similaridade.")

    uploaded_files = st.file_uploader("Escolha as imagens", accept_multiple_files=True, type=["jpg", "png", "jpeg"])

    if uploaded_files:
        os.makedirs("faces", exist_ok=True)

        for file in uploaded_files:
            with open(os.path.join("faces", file.name), "wb") as f:
                f.write(file.getbuffer())

        random.shuffle(uploaded_files)
        
        if st.button("Processar"):
            group_faces("faces")  
            st.success("Agrupamento realizado com sucesso!")

            st.write("Imagens agrupadas:")
            for i, label in enumerate(os.listdir("cluster")):
                st.subheader(f"Grupo {i + 1}")
                cluster_path = os.path.join("cluster", label)
                images_in_group = os.listdir(cluster_path)

                for image_file in images_in_group:
                    image_path = os.path.join(cluster_path, image_file)
                    st.image(image_path, caption=image_file, width=200)
