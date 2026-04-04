"""Flask application factory for the RAG Terminal app."""

import os
from flask import Flask
from flask_cors import CORS


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__, static_folder="static", static_url_path="")

    # Load config
    from app.config import Config
    app.config.from_object(Config)

    # Enable CORS
    CORS(app)

    # Ensure required directories exist
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["CHROMA_PERSIST_DIR"], exist_ok=True)

    # Register routes
    from app.routes import bp
    app.register_blueprint(bp)

    # Initialise the active vector store inside app context
    # (ChromaDB locally, Pinecone in hosted mode)
    with app.app_context():
        from app.ingestion.store import init_store
        init_store()

    # Serve index.html at root
    @app.route("/")
    def index():
        return app.send_static_file("index.html")

    return app
