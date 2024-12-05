# Visualização dos resultados
![view resultados](resultados.pdf)

# Estrutura do Projeto

O projeto está organizado nos seguintes arquivos:

- **main.py**: Contém a interface do usuário utilizando Streamlit. Permite o upload de imagens e aciona o processo de agrupamento.
- **clustering.py**: Implementa a lógica de agrupamento de rostos utilizando embeddings gerados pelo FaceNet e KMeans para clustering.
- **dirs.py**: Gerencia os diretórios para salvar os clusters de imagens gerados.
- **processing.py**: Realiza o pré-processamento das imagens, como redimensionamento e normalização.
- **requirements.txt**: Lista de dependências necessárias para executar o projeto.
- **test_face/**: Diretório contendo imagens de teste para verificar o funcionamento da aplicação.

## Funcionalidades

### Upload de Múltiplas Imagens
- Os usuários podem carregar várias imagens em formatos como JPG, PNG ou JPEG.

### Agrupamento de Rostos
- As imagens são agrupadas com base na similaridade facial, utilizando o modelo FaceNet para extração de embeddings e o algoritmo KMeans para clustering.

### Exibição dos Resultados
- Após o processamento, os clusters são exibidos com as imagens correspondentes.

### Pré-processamento de Imagens
- Imagens são redimensionadas para um tamanho fixo (160x160) e normalizadas para garantir a consistência.

## Como Executar

### Pré-requisitos

- Python 3.8 ou superior
- Pip instalado

### Passos para execução

1. **Clone este repositório**:

   ```bash
   git clone https://github.com/PegouOcodigoDev/clustering-de-images.git
   ```

2. **Crie e ative um ambiente virtual**:

   Crie um ambiente virtual para isolar as dependências do projeto:

   ```bash
   python -m venv env
   ```

   No Windows, ative o ambiente com:

   ```bash
   .\env\Scripts\activate
   ```

   No Linux ou Mac, ative o ambiente com:

   ```bash
   source env/bin/activate
   ```

3. **Instale as dependências**:

   Instale todas as dependências necessárias utilizando o arquivo `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

4. **Execute o script principal**:

   Para rodar a aplicação com a interface do usuário em Streamlit, execute o seguinte comando:

   ```bash
   streamlit run main.py
   ```

   Abra o navegador no endereço exibido no terminal (geralmente http://localhost:8501).

5. **Faça upload das imagens**:

   Faça upload das imagens do diretório `test_face/` ou de outras imagens de sua escolha.

6. **Clique no botão "Processar"** para iniciar o agrupamento.

7. **Visualize os grupos de imagens exibidos na interface**.

## Dependências Principais

- **Streamlit**: Para criar a interface do usuário.
- **FaceNet**: Para extração de embeddings faciais.
- **KMeans (Scikit-learn)**: Para realizar o agrupamento das imagens.
- **OpenCV**: Para pré-processamento das imagens.

## Estrutura de Diretórios

- **faces/**: Diretório temporário onde as imagens carregadas pelo usuário são salvas.
- **cluster/**: Diretório onde os clusters de imagens gerados são armazenados.
- **test_face/**: Contém imagens para testar a aplicação. Utilize este diretório para validar o agrupamento.

## Testes Unitários

O projeto inclui testes unitários para garantir que as funcionalidades principais estejam funcionando corretamente. Para rodar os testes, siga os passos abaixo:

1. **Certifique-se de estar no ambiente virtual ativado**.

2. **Instale as dependências de testes**:

   Se as dependências para testes não estiverem no `requirements.txt`, instale o `pytest` manualmente:

   ```bash
   pip install pytest
   ```

3. **Execute os testes**:

   Para rodar os testes, execute o seguinte comando:

   ```bash
   pytest
   ```

   Os testes estão localizados no diretório `tests/` e são organizados conforme o tipo de funcionalidade que está sendo testada (por exemplo, `test_clustering.py`, `test_dirs.py`, `test_processing.py`).

4. **Verifique o Resultado dos Testes**:

   O `pytest` irá mostrar no terminal se os testes passaram ou falharam, com detalhes sobre os erros, se houver.

### Observações

- O algoritmo embaralha as imagens antes do agrupamento para garantir que o processo não dependa da ordem de upload.
- Para garantir a precisão, o modelo espera imagens com resolução adequada e rostos visíveis.

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests.

## Licença

Este projeto está sob a licença MIT. Sinta-se à vontade para usá-lo e modificá-lo conforme necessário.
go.
