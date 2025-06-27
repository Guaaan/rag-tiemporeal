import http.client
import json
import os
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

VN_SERVER_ADDRESS = os.getenv('VN_SERVER_ADDRESS')
APP_KEY = os.getenv('APP_KEY')
APP_SECRET = os.getenv('APP_SECRET')
TYPE = os.getenv('TYPE', 'unifiedapi')
REDIRECT_URI = f'https://{VN_SERVER_ADDRESS}/ouath/token.php'
URI = f'https://{VN_SERVER_ADDRESS}/oauth/token.php'

# Configurar los parámetros de la solicitud
headers = {"Content-type": "application/x-www-form-urlencoded"}
content = (
    f"&client_id={APP_KEY}&client_secret={APP_SECRET}&grant_type=client_credentials&type={TYPE}&redirect_uri={REDIRECT_URI}"
)
conn = http.client.HTTPSConnection(VN_SERVER_ADDRESS)
conn.request("POST", URI, content, headers)

# Procesar la respuesta JSON
response = conn.getresponse()
response_data = response.read().decode('utf-8')
try:
    print(json.dumps(json.loads(response_data), indent=4))
except json.JSONDecodeError:
    print('Respuesta no es JSON válido:')
    print(response_data)

