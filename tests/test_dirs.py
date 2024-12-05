import os
import shutil
import pytest
from utils.dirs import make_groups_dir, del_group_dir

@pytest.fixture
def setup_directories():
    if not os.path.exists("test_cluster"):
        os.makedirs("test_cluster")
    yield
    shutil.rmtree("test_cluster", ignore_errors=True)

def test_del_group_dir(setup_directories):
    os.makedirs("test_cluster/test_subdir")
    del_group_dir("test_cluster")
    assert not os.path.exists("test_cluster"), "Diretório não foi excluído corretamente."

def test_make_groups_dir(setup_directories):
    images = [255 * i for i in range(10)]  
    labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  
    make_groups_dir(images, labels)
    cluster_dir = os.path.abspath("cluster")
    assert os.path.exists(cluster_dir), "Cluster directory não foi criado."
    for label in set(labels):
        assert os.path.exists(os.path.join(cluster_dir, f"cluster_{label}")), "Grupo não foi criado."
