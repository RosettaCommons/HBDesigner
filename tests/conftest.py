def pytest_addoption(parser):
    parser.addoption(
        "--device",
        action="store",
        default="gpu",
        choices=("cpu", "gpu"),
        help="Run example tests on cpu or gpu (default: gpu).",
    )
