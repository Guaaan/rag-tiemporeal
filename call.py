import requests

# — Paso 1: Obtener token —


def get_token():
    url = "https://api.n2p.io/v1/oauth/token/"
    payload = {
        "client_id": "6529351469236224",
        "username": "admin.ITAM",
        "password": "BxPH9kvQ",
        "grant_type": "password"
    }
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate, br",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    resp = requests.post(url, data=payload, headers=headers)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        print(f"❌ Error al obtener token: {e}")
        print(f"Status code: {resp.status_code}")
        print(f"Response: {resp.text}")
        raise
    return resp.json().get("access_token")

# — Paso 2: Realizar llamada con bearer token —


def make_call(access_token, from_number, to_number):
    url = "https://api.n2p.io/v1/calls/"
    headers = {
        "Authorization": f"Bearer {access_token}",
        'Content-Type': 'application/json; charset=utf-8',
        "Accept": "application/vnd.integrate.v1.9.0+json"
    }
    body = {
        "to": to_number,
        "from": from_number,
    }
    resp = requests.post(url, json=body, headers=headers)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        print("❌ Error al realizar la llamada:")
        print(f"Status code: {resp.status_code}")
        print(f"Response: {resp.text}")
        raise
    return resp.json()


if __name__ == "__main__":
    try:
        token = get_token()
        print("✅ Token obtenido:", token)

        result = make_call(token, "234802005", "945854758")
        print("📞 Llamada iniciada, respuesta API:", result)
    except requests.HTTPError as e:
        print("❌ Error en la ejecución. Consulta los detalles arriba.")
