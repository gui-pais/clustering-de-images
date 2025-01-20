import os
import shutil

def del_group_dir(dir_name):
    cluster_dir = os.path.abspath(dir_name)
    if os.path.exists(cluster_dir):
        try:
            shutil.rmtree(cluster_dir) 
            print(f"Diretório {cluster_dir} e seu conteúdo foram excluídos com sucesso.")
        except Exception as e:
            print(f"Erro ao excluir o diretório {cluster_dir}: {e}")
    else:
        print(f"O diretório {cluster_dir} não existe.")
