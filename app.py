import os

from dotenv import load_dotenv
from flask import Flask

from config import MAX_CONTENT_LENGTH
from db_helpers import ensure_app_dirs, init_db, seed_family_members
from services.embedding_service import create_profile_embedding_from_saved_photo

load_dotenv()

try:
    from deepface import DeepFace
except Exception as error:
    print("DeepFace not available:", error)
    DeepFace = None


def _register_legacy_endpoint_aliases(app):
    """Keep existing url_for('home') / url_for('review_page') calls working.

    Blueprint endpoints are normally named web.home, web.review_page, etc.
    This compatibility layer can be removed after templates are updated.
    """
    existing = set(app.view_functions)
    rules = list(app.url_map.iter_rules())

    for rule in rules:
        if not rule.endpoint.startswith("web."):
            continue

        legacy_endpoint = rule.endpoint.split(".", 1)[1]
        if legacy_endpoint in existing:
            continue

        methods = sorted(rule.methods - {"HEAD", "OPTIONS"})
        app.add_url_rule(
            rule.rule,
            endpoint=legacy_endpoint,
            view_func=app.view_functions[rule.endpoint],
            defaults=rule.defaults,
            methods=methods,
        )
        existing.add(legacy_endpoint)


def create_app():
    app = Flask(__name__)
    app.secret_key = os.getenv("FLASK_SECRET_KEY")
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
    app.config["JWT_SECRET_KEY"] = os.getenv(
        "JWT_SECRET_KEY",
        app.config["SECRET_KEY"],
    )
    app.config["EMBEDDING_ENCRYPTION_KEY"] = os.getenv(
        "EMBEDDING_ENCRYPTION_KEY"
    )
    app.config["DeepFace"] = DeepFace
    app.config["create_profile_embedding_from_saved_photo"] = (
        create_profile_embedding_from_saved_photo
    )

    ensure_app_dirs()

    from routes.api_auth import api_auth_bp
    from routes.api_members import api_members_bp
    from routes.api_profile_photos import api_profile_photos_bp
    from routes.admin import admin_bp
    from routes.telegram_api import telegram_api_bp
    from routes.telegram_web import telegram_web_bp
    from routes.web import web_bp

    app.register_blueprint(web_bp)
    app.register_blueprint(api_auth_bp)
    app.register_blueprint(api_members_bp)
    app.register_blueprint(api_profile_photos_bp)
    app.register_blueprint(telegram_api_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(telegram_web_bp)

    _register_legacy_endpoint_aliases(app)
    return app


app = create_app()


if __name__ == "__main__":
    init_db()
    seed_family_members()
    app.run(debug=True, host="0.0.0.0", port=5000)
