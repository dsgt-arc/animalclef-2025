from typer import Typer
from .preprocess.workflow import app as preprocess_app
from .embed.workflow import app as embed_app

app = Typer(no_args_is_help=True)
app.add_typer(preprocess_app, name="preprocess")
app.add_typer(embed_app, name="embed")
