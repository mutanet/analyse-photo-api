
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import requests
import os

app = FastAPI()

BASE_URL = "https://ton-api.onrender.com"  # à remplacer après déploiement
STATIC_DIR = "static"
CONTOURS_DIR = os.path.join(STATIC_DIR, "contours")
PLANS_DIR = os.path.join(STATIC_DIR, "plans")

os.makedirs(CONTOURS_DIR, exist_ok=True)
os.makedirs(PLANS_DIR, exist_ok=True)


def telecharger_image(url: str, path: str):
    r = requests.get(url)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)


def detecter_contour_et_trous(image_path: str):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_principal = max(contours, key=cv2.contourArea)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=5,
        maxRadius=200
    )

    positions_trous = []
    diametres_trous = []

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            positions_trous.append([int(x), int(y)])
            diametres_trous.append(int(2 * r))

    return contour_principal, positions_trous, diametres_trous, img.shape


def generer_svg_contour(id_piece: str, contour, positions_trous, diametres_trous, shape):
    h, w, _ = shape
    points = contour.squeeze()
    if len(points.shape) == 1:
        points = np.array([points])

    svg_path = os.path.join(CONTOURS_DIR, f"{id_piece}.svg")

    with open(svg_path, "w") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n')
        f.write('<g stroke="black" fill="none">\n')

        f.write('<polygon points="')
        for p in points:
            x, y = p
            f.write(f'{x},{y} ')
        f.write('" />\n')

        for (x, y), d in zip(positions_trous, diametres_trous):
            r = d / 2
            f.write(f'<circle cx="{x}" cy="{y}" r="{r}" />\n')

        f.write('</g>\n</svg>\n')

    return f"{BASE_URL}/static/contours/{id_piece}.svg"


def generer_svg_plan(id_piece: str, contour, positions_trous, diametres_trous, shape):
    h, w, _ = shape
    svg_path = os.path.join(PLANS_DIR, f"{id_piece}.svg")

    xs = [p[0][0] for p in contour]
    ys = [p[0][1] for p in contour]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    largeur = max_x - min_x
    hauteur = max_y - min_y

    with open(svg_path, "w") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n')
        f.write('<g stroke="black" fill="none" font-size="12">\n')

        f.write(f'<rect x="{min_x}" y="{min_y}" width="{largeur}" height="{hauteur}" />\n')

        for (x, y), d in zip(positions_trous, diametres_trous):
            r = d / 2
            f.write(f'<circle cx="{x}" cy="{y}" r="{r}" />\n')

        f.write(f'<text x="{min_x}" y="{min_y - 10}">Largeur: {largeur}</text>\n')
        f.write(f'<text x="{max_x + 10}" y="{min_y + hauteur/2}">Hauteur: {hauteur}</text>\n')

        f.write('</g>\n</svg>\n')

    return f"{BASE_URL}/static/plans/{id_piece}.svg"


@app.post("/analyse-photo")
def analyse_photo(payload: dict = Body(...)):
    id_piece = payload.get("id_piece")
    photo_url = payload.get("photo_url")

    if not id_piece or not photo_url:
        return JSONResponse({"error": "id_piece ou photo_url manquant"}, status_code=400)

    image_path = os.path.join(STATIC_DIR, f"{id_piece}.jpg")
    telecharger_image(photo_url, image_path)

    contour, positions_trous, diametres_trous, shape = detecter_contour_et_trous(image_path)
    contour_svg_url = generer_svg_contour(id_piece, contour, positions_trous, diametres_trous, shape)

    return JSONResponse({
        "contour_svg_url": contour_svg_url,
        "nb_trous": len(positions_trous),
        "positions_trous": positions_trous,
        "diametres_trous": diametres_trous,
        "shape": shape,
        "contour_points": contour.squeeze().tolist()
    })


@app.post("/generer-plan")
def generer_plan(payload: dict = Body(...)):
    id_piece = payload.get("id_piece")
    positions_trous = payload.get("positions_trous", [])
    diametres_trous = payload.get("diametres_trous", [])
    contour_points = payload.get("contour_points", None)
    shape = payload.get("shape", [500, 500, 3])

    if contour_points is None:
        return JSONResponse({"error": "contour_points manquant"}, status_code=400)

    contour = np.array(contour_points, dtype=np.int32).reshape(-1, 1, 2)
    shape_tuple = (shape[0], shape[1], shape[2])

    plan_svg_url = generer_svg_plan(id_piece, contour, positions_trous, diametres_trous, shape_tuple)

    return JSONResponse({
        "plan_svg_url": plan_svg_url
    })
