from QRC.NQS import hello


def test_hello() -> None:
    assert hello() == "Hello from QRC.NQS!"
