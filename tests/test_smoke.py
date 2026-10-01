from deployguard.smoke import healthcheck


def test_healthcheck() -> None:
    assert healthcheck() == "deployguard-ok"
