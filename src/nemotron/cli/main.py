from nemotron.cli.common import make_typer
from nemotron.cli.data import app as data_app
from nemotron.cli.rrcg import app as rrcg_app

app = make_typer()
app.add_typer(data_app, name='data')
app.add_typer(rrcg_app, name='rrcg')
