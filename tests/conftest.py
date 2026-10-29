import pytest


@pytest.fixture(autouse=True)
def spine_license_confirmed(monkeypatch):
    """The tests pose synthetic skeletons with the vendored Spine runtime on behalf of a licensed developer."""
    monkeypatch.setenv('SPINE_ARTIST_SPINE_LICENSE', 'confirmed')
