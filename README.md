# Family Focus — Server AI Version

This version uses Flask for server-side face recognition.

## Install

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## Windows note

`face-recognition` depends on `dlib`, which can be difficult to install on Windows.
If install fails, try:

```bash
pip install cmake
pip install dlib
pip install face-recognition
```

Python 3.10 or 3.11 is usually easier than 3.12 for this package.

## iPhone test

Camera needs HTTPS. Run Flask and expose it:

```bash
cloudflared tunnel --url http://localhost:5000
```

Open the HTTPS URL on your iPhone.
