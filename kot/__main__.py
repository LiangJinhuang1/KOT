"""Allow ``python -m kot`` to invoke the training CLI."""

from src.training.runner import main


if __name__ == "__main__":
    main()
