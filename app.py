import logging
import os
import tempfile
import uuid
import zipfile

from flask import Flask, Blueprint, render_template, request, redirect, session, abort, flash
from flask import jsonify, send_file, send_from_directory, url_for, current_app
from flask_sqlalchemy import SQLAlchemy
from PIL import Image as PILImage
from ultralytics import YOLO
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY") or "your-super-secret-key-change-me"
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL") or (
        "sqlite:///" + os.path.join(basedir, "instance", "supersus.db")
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = os.path.join(basedir, "uploads")
    MODEL_FOLDER = os.path.join(basedir, "models")
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "pt"}


app = Flask(__name__)
app.config.from_object(Config)

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["MODEL_FOLDER"], exist_ok=True)
os.makedirs("instance", exist_ok=True)

db = SQLAlchemy()


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    model_path = db.Column(db.String, nullable=True)
    images = db.relationship("Image", backref="project", lazy=True)
    labels = db.relationship(
        "ProjectLabel",
        backref="project",
        lazy=True,
        cascade="all, delete-orphan",
        order_by="ProjectLabel.label_index.asc()",
    )


class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    project_id = db.Column(db.Integer, db.ForeignKey("project.id"))
    original_path = db.Column(db.String, nullable=True)
    annotations = db.relationship("Annotation", backref="image", lazy=True)


class Annotation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_id = db.Column(db.Integer, db.ForeignKey("image.id"))
    x = db.Column(db.Float)
    y = db.Column(db.Float)
    w = db.Column(db.Float)
    h = db.Column(db.Float)
    label = db.Column(db.Integer)


class ProjectLabel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey("project.id"), nullable=False)
    label_index = db.Column(db.Integer, nullable=False)
    name = db.Column(db.String(120), nullable=False)


db.init_app(app)

auth_bp = Blueprint("auth", __name__)
project_bp = Blueprint("project", __name__)
annotate_bp = Blueprint("annotate", __name__)
export_bp = Blueprint("export", __name__)

@app.route("/favicon.ico")
def favicon():
    return "", 204

# Cache models per project so repeat inference stays fast.
model_cache = {}


def allowed_file(filename, allowed_extensions=None):
    allowed_extensions = allowed_extensions or current_app.config["ALLOWED_EXTENSIONS"]
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def make_unique_upload_name(filename):
    safe_name = secure_filename(filename)
    stem, ext = os.path.splitext(safe_name)
    return f"{stem}_{uuid.uuid4().hex[:8]}{ext}"


def get_model_for_project(project_id: int) -> YOLO:
    if project_id in model_cache:
        return model_cache[project_id]

    project = db.session.get(Project, project_id)
    if not project:
        raise ValueError("Project not found")

    model_path = project.model_path
    if not model_path or not os.path.exists(model_path):
        default_path = os.path.join(current_app.config["MODEL_FOLDER"], "best.pt")
        if os.path.exists(default_path):
            model_path = default_path
        else:
            logging.warning("No model for project %s and no default best.pt", project_id)
            raise FileNotFoundError("No YOLO model available")

    try:
        model = YOLO(model_path)
        logging.info("Loaded YOLO model for project %s", project_id)
        model_cache[project_id] = model
        return model
    except Exception as e:
        logging.error("Failed to load model %s: %s", model_path, e)
        raise


def _get_user_image(image_id):
    image = db.session.get(Image, image_id)
    if not image:
        abort(404)
    if image.user_id != session.get("user_id"):
        abort(403)
    return image


def _get_user_project(project_id):
    project = db.session.get(Project, project_id)
    if not project:
        abort(404)
    if project.user_id != session.get("user_id"):
        abort(403)
    return project


def _ensure_project_labels(project_id):
    labels = ProjectLabel.query.filter_by(project_id=project_id).order_by(ProjectLabel.label_index.asc()).all()
    if labels:
        return labels

    default_label = ProjectLabel(project_id=project_id, label_index=0, name="Class 0")
    db.session.add(default_label)
    db.session.commit()
    return [default_label]


def _get_label_map(project_id):
    return {label.label_index: label.name for label in _ensure_project_labels(project_id)}


def _delete_image_files(image_obj):
    file_candidates = {
        image_obj.original_path,
        os.path.join(current_app.config["UPLOAD_FOLDER"], image_obj.filename),
    }
    for file_path in file_candidates:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                logging.warning("Could not remove image file %s", file_path)


def _resolve_image_path(image_obj):
    file_candidates = [
        image_obj.original_path,
        os.path.join(current_app.config["UPLOAD_FOLDER"], image_obj.filename),
    ]
    for file_path in file_candidates:
        if file_path and os.path.exists(file_path):
            return file_path
    return None


def _build_image_url(image_obj):
    image_url = url_for("annotate.image_file", image_id=image_obj.id, _external=True)
    return image_url.replace("://0.0.0.0", "://127.0.0.1", 1)


def _build_upload_url(filename):
    upload_url = url_for("annotate.uploaded_file", filename=filename, _external=True)
    return upload_url.replace("://0.0.0.0", "://127.0.0.1", 1)


def _delete_project_resources(project):
    for image in list(project.images):
        Annotation.query.filter_by(image_id=image.id).delete()
        _delete_image_files(image)
        db.session.delete(image)

    if project.model_path and os.path.exists(project.model_path):
        try:
            os.remove(project.model_path)
        except OSError:
            logging.warning("Could not remove model file %s", project.model_path)

    ProjectLabel.query.filter_by(project_id=project.id).delete()
    model_cache.pop(project.id, None)
    db.session.delete(project)


def _run_detection_for_image(image_obj):
    img_path = image_obj.original_path or os.path.join(
        current_app.config["UPLOAD_FOLDER"], image_obj.filename
    )
    if not os.path.exists(img_path):
        raise FileNotFoundError("Image file not found on disk")

    pil_image = PILImage.open(img_path).convert("RGB")
    model = get_model_for_project(image_obj.project_id)
    results = model.predict(
        pil_image,
        conf=0.1,
        iou=0.1,
        verbose=False,
    )

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                {
                    "x": x1,
                    "y": y1,
                    "w": x2 - x1,
                    "h": y2 - y1,
                    "label": int(box.cls.item()),
                }
            )
    return detections


def _replace_annotations(image_id, boxes):
    Annotation.query.filter_by(image_id=image_id).delete()
    for box in boxes:
        db.session.add(
            Annotation(
                image_id=image_id,
                x=float(box["x"]),
                y=float(box["y"]),
                w=float(box["w"]),
                h=float(box["h"]),
                label=int(box.get("label", 0)),
            )
        )


def _annotation_to_yolo_line(annotation, image_width, image_height):
    x = max(0.0, min(float(annotation.x), float(image_width)))
    y = max(0.0, min(float(annotation.y), float(image_height)))
    w = max(0.0, min(float(annotation.w), float(image_width - x)))
    h = max(0.0, min(float(annotation.h), float(image_height - y)))

    x_center = (x + (w / 2.0)) / float(image_width)
    y_center = (y + (h / 2.0)) / float(image_height)
    norm_w = w / float(image_width)
    norm_h = h / float(image_height)

    return f"{int(annotation.label)} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"


@auth_bp.route("/", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session["user_id"] = user.id
            return redirect("/dashboard")
        error = "Invalid username or password"
    return render_template("login.html", error=error)


@auth_bp.route("/signup", methods=["GET", "POST"])
def signup():
    error = None
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if User.query.filter_by(username=username).first():
            error = "Username already exists"
        else:
            user = User(username=username)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            session["user_id"] = user.id
            return redirect("/dashboard")
    return render_template("signup.html", error=error)


@auth_bp.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect("/")
    projects = Project.query.filter_by(user_id=session["user_id"]).all()
    return render_template("dashboard.html", projects=projects)


@project_bp.route("/project/create", methods=["POST"])
def create_project():
    if "user_id" not in session:
        abort(401)
    project = Project(name=request.form["name"], user_id=session["user_id"])
    db.session.add(project)
    db.session.commit()
    return redirect(f"/project/{project.id}")


@project_bp.route("/project/<int:project_id>")
def view_project(project_id):
    if "user_id" not in session:
        abort(401)

    project = _get_user_project(project_id)
    model_filename = os.path.basename(project.model_path) if project.model_path else None
    project_images = Image.query.filter_by(project_id=project.id).order_by(Image.id.asc()).all()
    labels = _ensure_project_labels(project.id)
    image_urls = {image.id: _build_image_url(image) for image in project_images}
    upload_urls = {image.id: _build_upload_url(image.filename) for image in project_images}
    return render_template(
        "project.html",
        project=project,
        model_filename=model_filename,
        project_images=project_images,
        labels=labels,
        image_urls=image_urls,
        upload_urls=upload_urls,
    )


@project_bp.route("/project/<int:project_id>/upload", methods=["POST"])
def upload_image(project_id):
    if "user_id" not in session:
        abort(401)

    _get_user_project(project_id)
    files = request.files.getlist("files")
    if not files:
        single_file = request.files.get("file")
        if single_file:
            files = [single_file]
    if not files:
        return "No files", 400

    uploaded_count = 0
    for file in files:
        if not file or not file.filename:
            continue
        if not allowed_file(file.filename, {"png", "jpg", "jpeg"}):
            continue

        filename = make_unique_upload_name(file.filename)
        path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
        file.save(path)
        db.session.add(
            Image(
                filename=filename,
                user_id=session["user_id"],
                project_id=project_id,
                original_path=path,
            )
        )
        uploaded_count += 1

    if uploaded_count == 0:
        return "No valid image files", 400

    db.session.commit()
    return redirect(f"/project/{project_id}")


@project_bp.route("/project/<int:project_id>/upload_model", methods=["POST"])
def upload_model(project_id):
    if "user_id" not in session:
        abort(401)

    project = _get_user_project(project_id)
    file = request.files.get("model")
    if not file:
        return "No file", 400

    filename = f"project_{project_id}.pt"
    path = os.path.join(current_app.config["MODEL_FOLDER"], filename)
    file.save(path)
    project.model_path = path
    db.session.commit()
    model_cache.pop(project_id, None)
    return redirect(f"/project/{project_id}")


@project_bp.route("/project/<int:project_id>/labels", methods=["POST"])
def add_project_label(project_id):
    if "user_id" not in session:
        abort(401)

    project = _get_user_project(project_id)
    label_name = (request.form.get("label_name") or "").strip()
    if not label_name:
        flash("Label name cannot be empty.", "danger")
        return redirect(f"/project/{project_id}")

    existing_labels = _ensure_project_labels(project.id)
    existing_names = {label.name.lower() for label in existing_labels}
    if label_name.lower() in existing_names:
        flash("That label name already exists in this project.", "warning")
        return redirect(f"/project/{project_id}")

    next_index = max((label.label_index for label in existing_labels), default=-1) + 1
    db.session.add(ProjectLabel(project_id=project.id, label_index=next_index, name=label_name))
    db.session.commit()
    flash(f"Added label '{label_name}' as class {next_index}.", "success")
    return redirect(f"/project/{project_id}")


@project_bp.route("/project/<int:project_id>/labels/<int:label_id>/delete", methods=["POST"])
def delete_project_label(project_id, label_id):
    if "user_id" not in session:
        abort(401)

    project = _get_user_project(project_id)
    label = ProjectLabel.query.get_or_404(label_id)
    if label.project_id != project.id:
        abort(404)

    annotation_count = (
        db.session.query(Annotation)
        .join(Image, Annotation.image_id == Image.id)
        .filter(Image.project_id == project.id, Annotation.label == label.label_index)
        .count()
    )
    if annotation_count > 0:
        flash("This label is already used in annotations, so it cannot be deleted yet.", "warning")
        return redirect(f"/project/{project_id}")

    db.session.delete(label)
    db.session.commit()
    flash(f"Deleted label '{label.name}'.", "success")
    return redirect(f"/project/{project_id}")


@project_bp.route("/project/<int:project_id>/auto_annotate", methods=["POST"])
def auto_annotate_project(project_id):
    if "user_id" not in session:
        abort(401)

    project = _get_user_project(project_id)
    images = Image.query.filter_by(project_id=project.id).order_by(Image.id.asc()).all()
    if not images:
        return redirect(f"/project/{project_id}")

    try:
        for image in images:
            detections = _run_detection_for_image(image)
            _replace_annotations(image.id, detections)
        db.session.commit()
    except FileNotFoundError as e:
        db.session.rollback()
        flash(str(e) + ". Upload a YOLO model first or add models/best.pt.", "warning")
        return redirect(f"/project/{project_id}")
    except Exception as e:
        db.session.rollback()
        logging.exception("Project auto-annotation failed")
        flash("Auto-annotate failed. Check the server logs for details.", "danger")
        return redirect(f"/project/{project_id}")

    return redirect(url_for("annotate.annotate", image_id=images[0].id))


@project_bp.route("/project/<int:project_id>/images/<int:image_id>/delete", methods=["POST"])
def delete_project_image(project_id, image_id):
    if "user_id" not in session:
        abort(401)

    project = _get_user_project(project_id)
    image = _get_user_image(image_id)
    if image.project_id != project.id:
        abort(404)

    project_images = Image.query.filter_by(project_id=project.id).order_by(Image.id.asc()).all()
    image_ids = [project_image.id for project_image in project_images]
    current_index = image_ids.index(image.id)
    next_image = project_images[current_index + 1] if current_index < len(project_images) - 1 else None
    prev_image = project_images[current_index - 1] if current_index > 0 else None

    Annotation.query.filter_by(image_id=image.id).delete()
    _delete_image_files(image)
    db.session.delete(image)
    db.session.commit()
    flash(f"Deleted image '{image.filename}'.", "success")

    redirect_target = request.form.get("next")
    if redirect_target == "annotate":
        if next_image:
            return redirect(url_for("annotate.annotate", image_id=next_image.id))
        if prev_image:
            return redirect(url_for("annotate.annotate", image_id=prev_image.id))

    return redirect(f"/project/{project_id}")


@project_bp.route("/project/<int:project_id>/delete", methods=["POST"])
def delete_project(project_id):
    if "user_id" not in session:
        abort(401)

    project = _get_user_project(project_id)
    project_name = project.name
    _delete_project_resources(project)
    db.session.commit()
    flash(f"Deleted project '{project_name}'.", "success")
    return redirect("/dashboard")


@annotate_bp.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(current_app.config["UPLOAD_FOLDER"], filename)


@annotate_bp.route("/image/<int:image_id>")
def image_file(image_id):
    image = db.session.get(Image, image_id)
    if not image:
        abort(404)
    image_path = _resolve_image_path(image)
    if not image_path:
        abort(404)

    response = send_file(image_path, conditional=True, max_age=0)
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


@annotate_bp.route("/annotate/<int:image_id>")
def annotate(image_id):
    if "user_id" not in session:
        abort(401)

    image = _get_user_image(image_id)
    project_images = Image.query.filter_by(project_id=image.project_id).order_by(Image.id.asc()).all()
    labels = _ensure_project_labels(image.project_id)
    image_url = _build_image_url(image)
    upload_fallback_url = _build_upload_url(image.filename)
    image_ids = [project_image.id for project_image in project_images]
    current_index = image_ids.index(image.id)
    prev_image = project_images[current_index - 1] if current_index > 0 else None
    next_image = project_images[current_index + 1] if current_index < len(project_images) - 1 else None

    return render_template(
        "annotate.html",
        image=image,
        image_url=image_url,
        upload_fallback_url=upload_fallback_url,
        project_images=project_images,
        prev_image=prev_image,
        next_image=next_image,
        current_index=current_index,
        labels=labels,
    )


@annotate_bp.route("/load/<int:image_id>")
def load(image_id):
    if "user_id" not in session:
        abort(401)
    _get_user_image(image_id)
    anns = Annotation.query.filter_by(image_id=image_id).all()
    return jsonify(
        [{"x": a.x, "y": a.y, "w": a.w, "h": a.h, "label": a.label} for a in anns]
    )


@annotate_bp.route("/save/<int:image_id>", methods=["POST"])
def save(image_id):
    if "user_id" not in session:
        abort(401)

    _get_user_image(image_id)
    data = request.get_json() or []
    _replace_annotations(image_id, data)
    db.session.commit()
    return {"status": "ok"}


@annotate_bp.route("/detect/<int:image_id>")
def detect(image_id):
    if "user_id" not in session:
        abort(401)

    image_obj = _get_user_image(image_id)
    try:
        detections = _run_detection_for_image(image_obj)
        valid_label_ids = set(_get_label_map(image_obj.project_id).keys())
        for detection in detections:
            if detection["label"] not in valid_label_ids:
                detection["label"] = 0
        return jsonify(detections)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logging.exception("YOLO inference error")
        return jsonify(
            {"error": str(e), "message": "Detection failed - check server logs"}
        ), 500


@export_bp.route("/export")
def export():
    if "user_id" not in session:
        return redirect("/")

    user_id = session["user_id"]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    zip_path = tmp.name
    tmp.close()

    with zipfile.ZipFile(zip_path, "w") as z:
        images = Image.query.filter_by(user_id=user_id).all()
        for img in images:
            img_path = img.original_path or os.path.join(
                current_app.config["UPLOAD_FOLDER"], img.filename
            )
            if os.path.exists(img_path):
                z.write(img_path, arcname=img.filename)
                with PILImage.open(img_path) as pil_image:
                    image_width, image_height = pil_image.size
            else:
                continue

            anns = Annotation.query.filter_by(image_id=img.id).all()
            txt_name = img.filename.rsplit(".", 1)[0] + ".txt"
            lines = [
                _annotation_to_yolo_line(a, image_width, image_height)
                for a in anns
            ]
            txt_path = os.path.join(tempfile.gettempdir(), txt_name)
            with open(txt_path, "w") as f:
                f.write("\n".join(lines))
            z.write(txt_path, arcname=txt_name)
            os.remove(txt_path)

    return send_file(zip_path, as_attachment=True)


app.register_blueprint(auth_bp)
app.register_blueprint(project_bp)
app.register_blueprint(annotate_bp)
app.register_blueprint(export_bp)

with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
