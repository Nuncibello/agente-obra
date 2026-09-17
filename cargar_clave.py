"""
Toma la clave de Gemini del portapapeles (no hay que pegarla en ningún lado), la prueba,
la guarda en .env (que está en .gitignore) y limpia el portapapeles.

  1. En Google Cloud tocá "Mostrar clave" y copiala.
  2. .venv\\Scripts\\python cargar_clave.py
"""

import os
import subprocess
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))


def portapapeles():
    r = subprocess.run(["powershell", "-NoProfile", "-Command", "Get-Clipboard"], capture_output=True, text=True)
    return (r.stdout or "").strip()


def limpiar_portapapeles():
    subprocess.run(["powershell", "-NoProfile", "-Command", "Set-Clipboard -Value ' '"], capture_output=True)


clave = portapapeles()
if " " in clave or len(clave) < 30 or not (clave.startswith("AIza") or clave.startswith("AQ.")):
    sys.exit("En el portapapeles no hay una clave de Gemini. Tocá 'Mostrar clave', copiala y volvé a correr esto.")

import logging

logging.disable(logging.WARNING)
from google import genai

modo, errores = None, []
for vertex in (False, True):  # las claves nuevas (AQ.) pueden ser de Gemini API o de Vertex en modo express
    cliente = genai.Client(api_key=clave, vertexai=vertex)  # guardarlo en una variable: si no, se cierra antes de usarlo
    try:
        r = cliente.models.generate_content(model="gemini-2.5-flash", contents="Respondé solo: ok")
        modo = vertex
        print("FUNCIONA:", (r.text or "").strip()[:20], "(Vertex AI)" if vertex else "(Gemini API)")
        break
    except Exception as e:
        errores.append(("Vertex" if vertex else "Gemini API") + ": " + str(e)[:160])
ultimo = " | ".join(errores)

if modo is None:
    sys.exit(f"La clave no funcionó: {ultimo}")

with open(os.path.join(AQUI, ".env"), "w", encoding="utf-8") as f:
    f.write(f"GOOGLE_API_KEY={clave}\nGOOGLE_GENAI_USE_VERTEXAI={'TRUE' if modo else 'FALSE'}\n")
limpiar_portapapeles()
print("LISTO: clave guardada en .env y portapapeles limpio. Avisale a Claude.")
