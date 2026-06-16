from functools import wraps
from flask import session, redirect, url_for, flash

def login_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in first.")
            return redirect(url_for("login"))
        return func(*args, **kwargs)

    return wrapper


def get_current_family_id():
    return session.get("family_id")


def get_current_user_id():
    return session.get("user_id")
    