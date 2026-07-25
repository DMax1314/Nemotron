from nemotron.cli.common import make_typer
from nemotron.cli.data import app as data_app

app = make_typer()
app.add_typer(data_app, name='data')
