"""Package entry point. It forwards command-line execution to the thin CLI layer."""

from .cli import main


if __name__ == "__main__":
    main()