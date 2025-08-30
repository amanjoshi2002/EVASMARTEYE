import os
import threading
import time
from flask import Flask, render_template, redirect, url_for, request, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from werkzeug.security import generate_password_hash, check_password_hash

from src import detection

app = Flask(__name__)
app.config["SECRET_KEY"] = "your_secret_key_here"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///cameras.db"
db = SQLAlchemy(app)

login_manager = LoginManager(app)
login_manager.login_view = "login"

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))

class Camera(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150))
    ip = db.Column(db.String(100))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = User.query.filter_by(username=request.form["username"]).first()
        if user and check_password_hash(user.password, request.form["password"]):
            login_user(user)
            return redirect(url_for("cameras"))
        flash("Invalid credentials")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

@app.route("/")
@login_required
def cameras():
    cams = Camera.query.all()
    return render_template("cameras.html", cameras=cams)

@app.route("/add", methods=["POST"])
@login_required
def add_camera():
    name = request.form["name"]
    ip = request.form["ip"]
    db.session.add(Camera(name=name, ip=ip))
    db.session.commit()
    return redirect(url_for("cameras"))

@app.route("/edit/<int:camera_id>", methods=["GET", "POST"])
@login_required
def edit_camera(camera_id):
    cam = Camera.query.get_or_404(camera_id)
    if request.method == "POST":
        cam.name = request.form["name"]
        cam.ip = request.form["ip"]
        db.session.commit()
        return redirect(url_for("cameras"))
    return render_template("edit_camera.html", camera=cam)

@app.route("/delete/<int:camera_id>")
@login_required
def delete_camera(camera_id):
    cam = Camera.query.get_or_404(camera_id)
    db.session.delete(cam)
    db.session.commit()
    return redirect(url_for("cameras"))

def run_detection():
    with app.app_context():
        cameras = Camera.query.all()
        detection.run_detection(cameras)
		
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(username="admin").first():
            db.session.add(User(username="admin", password=generate_password_hash("admin123")))
            db.session.commit()
    threading.Thread(target=run_detection, daemon=True).start()
    app.run(host="0.0.0.0", port=5001, debug=True)