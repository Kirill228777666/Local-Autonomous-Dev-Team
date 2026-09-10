from autodev.runtime import free_port

def test_free_port_is_bindable() -> None:
    assert 0 < free_port() < 65536
