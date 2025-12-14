from clustering.registry import get_clusterer


def test_get_clusterer_optics():
    c = get_clusterer("optics")
    assert c.name == "optics"


def test_get_clusterer_dbscan():
    c = get_clusterer("dbscan")
    assert c.name == "dbscan"
