from typer import Typer


def make_typer() -> Typer:
    """Creates a Typer instance with common settings for all CLI commands.

    This function returns a Typer instance configured with the following
    settings:

        - Disable Typer’s automatic shell-completion command generation.
        - Disable Rich markup in help strings.
        - Enable suggestions for mistyped commands.
        - Disable Typer’s pretty exception formatting.

    Returns:
        A Typer instance with predefined settings for the CLI application.
    """

    return Typer(
        add_completion=False,
        rich_markup_mode=None,
        suggest_commands=True,
        pretty_exceptions_enable=False,
    )
