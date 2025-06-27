# call.py
import requests
import webbrowser
from fastapi import FastAPI, Request
from threading import Thread
from urllib.parse import urlencode
from uvicorn import run

CLIENT_ID = "11X0j7Wp"
CLIENT_SECRET = "avrkhx5ZaVxDiMBbX"
REDIRECT_URI = "http://localhost:9000/callback"
AUTH_URL = "https://auth.net2phone.com/connect/authorize"
TOKEN_URL = "https://auth.net2phone.com/connect/token"
SCOPE = "uapi"

code_holder = {"code": None}
app = FastAPI()

@app.get("/callback")
async def callback(request: Request):
    code_holder["code"] = request.query_params.get("code")
    return "Token recibido. Puedes cerrar esta ventana."

def start_fastapi():
    run(app, host="0.0.0.0", port=9000)

def authenticate():
    # Iniciar servidor
    Thread(target=start_fastapi, daemon=True).start()
    # Abrir navegador
    params = { "response_type": "code", "client_id": CLIENT_ID,
               "redirect_uri": REDIRECT_URI, "scope": SCOPE }
    webbrowser.open(f"{AUTH_URL}?{urlencode(params)}")
    # Esperar el code...
    import time
    for _ in range(60):
        if code_holder["code"]:
            break
        time.sleep(1)
    code = code_holder["code"]
    if not code:
        raise Exception("Timeout en autenticación")
    # Intercambiar token
    resp = requests.post(TOKEN_URL, data={
        "grant_type":"authorization_code", "code":code,
        "redirect_uri":REDIRECT_URI, "client_id":CLIENT_ID,
        "client_secret":CLIENT_SECRET
    })
    resp.raise_for_status()
    return resp.json().get("access_token")
