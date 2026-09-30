def detecter_contour_et_trous(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Impossible de lire l'image.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Trouver TOUS les contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Le contour principal = celui avec la plus grande surface
    contour_principal = max(contours, key=cv2.contourArea)

    positions_trous = []
    diametres_trous = []

    # Parcourir tous les contours internes
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)

        # Filtrer les petits contours (bruit)
        if area < 200:  
            continue

        # Filtrer le contour principal
        if np.array_equal(cnt, contour_principal):
            continue

        # Calcul du centre du trou
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Diamètre estimé à partir de l'aire (même si ellipse)
        diametre = int(np.sqrt(area / np.pi) * 2)

        positions_trous.append([cx, cy])
        diametres_trous.append(diametre)

    return contour_principal, positions_trous, diametres_trous, img.shape
