def test_import():
    try:
        import constrained_nn_eq_discovery
    except ImportError:
        assert False, "Failed to import constrained_nn_eq_discovery"
    assert True
