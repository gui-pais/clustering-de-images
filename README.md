# Estrutura do Projeto

O projeto está organizado nos seguintes arquivos:

- **`main.py`**: Contém a interface do usuário utilizando Streamlit. Permite o upload de imagens e aciona o processo de agrupamento.
- **`clustering.py`**: Implementa a lógica de agrupamento de rostos utilizando embeddings gerados pelo FaceNet e KMeans para clustering.
- **`dirs.py`**: Gerencia os diretórios para salvar os clusters de imagens gerados.
- **`processing.py`**: Realiza o pré-processamento das imagens, como redimensionamento e normalização.
- **`requirements.txt`**: Lista de dependências necessárias para executar o projeto.
- **`test_face/`**: Diretório contendo imagens de teste para verificar o funcionamento da aplicação.

## Funcionalidades

1. **Upload de Múltiplas Imagens**:
   - Os usuários podem carregar várias imagens em formatos como JPG, PNG ou JPEG.
   
2. **Agrupamento de Rostos**:
   - As imagens são agrupadas com base na similaridade facial, utilizando o modelo FaceNet para extração de embeddings e o algoritmo KMeans para clustering.

3. **Exibição dos Resultados**:
   - Após o processamento, os clusters são exibidos com as imagens correspondentes.

4. **Pré-processamento de Imagens**:
   - Imagens são redimensionadas para um tamanho fixo (160x160) e normalizadas para garantir a consistência.

## Como Executar

### Pré-requisitos

- Python 3.8 ou superior
- Pip instalado
- Instalação de dependências listadas no arquivo `requirements.txt`

### Passos para execução

1. Clone este repositório:
   ```bash
   git clone <url_do_repositorio>
   cd <nome_do_repositorio>
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

3. Execute o script principal:
   ```bash
   streamlit run main.py
   ```

4. Abra o navegador no endereço exibido no terminal (geralmente `http://localhost:8501`).

5. Faça upload das imagens do diretório `test_face/` ou de outras imagens de sua escolha.

6. Clique no botão "Processar" para iniciar o agrupamento.

7. Visualize os grupos de imagens exibidos na interface.

## Dependências Principais

- **Streamlit**: Para criar a interface do usuário.
- **FaceNet**: Para extração de embeddings faciais.
- **KMeans (Scikit-learn)**: Para realizar o agrupamento das imagens.
- **OpenCV**: Para pré-processamento das imagens.

## Estrutura de Diretórios

- **`faces/`**: Diretório temporário onde as imagens carregadas pelo usuário são salvas.
- **`cluster/`**: Diretório onde os clusters de imagens gerados são armazenados.

## Diretório de Testes

- **`test_face/`**: Contém imagens para testar a aplicação. Utilize este diretório para validar o agrupamento.

## Observações

- O algoritmo embaralha as imagens antes do agrupamento para garantir que o processo não dependa da ordem de upload.
- Para garantir a precisão, o modelo espera imagens com resolução adequada e rostos visíveis.

## Resultados
![resultados](resultados.pdf)


## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests.

## Licença

Este projeto está sob a licença MIT. Sinta-se à vontade para usá-lo e modificá-lo conforme necessário.

---

Divirta-se explorando o agrupamento de rostos! 🎉
