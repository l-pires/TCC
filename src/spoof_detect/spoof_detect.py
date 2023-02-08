import cv2
from matplotlib import cm, pyplot as plt


face_cascade = cv2.CascadeClassifier('src/assets/haarcascade_frontalface_default.xml')


def get_face(gray:cv2.Mat):
    """
    Aplica o classificador 'haarcascade_frontalface_default.xml' na imagem passada
    e retorna a face detectada com maior área total da imagem.

    ### Params
        gray : cv2.Mat
            Imagem openCV (cv2.Mat) de canal único

    ### Returns
        None | cv2.Mat
            A imagem recortada na região da face , se encontrada
    """
    max_area = 0
    x0 = y0 = w0 = h0 = 0
    
    for (x, y, w, h) in face_cascade.detectMultiScale(gray, 1.25, 4):
        area = w * h
        if area > max_area:
            max_area = area
            (x0, y0, w0, h0) = (x, y, w, h)
            
    if max_area == 0:
        return None
    # cv2.rectangle(gray, (x0, y0), (x0+w0, y0+h0), (0, 0, 255), 2)
    return gray[y0:y0 + h0, x0:x0 + w0]


def save_fig(imgs:tuple[cv2.Mat], path:str=None):
    """
    Salva as imagens passadas em uma figura no caminho informado.
    Imagens: (Imagem original, imagem monocromática, imagem da 
    face recortada e imagem binária aplicado o limite)

    ### Params
        imgs : tuple[cv2.Mat]
            4 Imagens openCV (cv2.Mat),
            sendo a primeira de 3 canais
            e as demais de canal único
        ?path : str
            Caminho para o arquivo
    """
    (img, gray, face, thresh) = imgs

    fig, ax = plt.subplots(2, 2)
    for x in ax:
        for y in x:
            y.set_axis_off()
    ax[0,0].imshow(img[...,::-1])
    ax[0,1].imshow(gray, cmap=cm.gray)
    ax[1,0].imshow(face, cmap=cm.gray)
    ax[1,1].imshow(thresh, cmap=cm.gray)

    if not path is None:
        fig.savefig(path)

    plt.close()


def reflection_detect(img:cv2.Mat, threshold:int=222) -> tuple[int, tuple[cv2.Mat]]:
    """
    A partir da imagem recebida, executa a rotina de detecção de reflexão,
    passando pela detecção facial, suavização da imagem, e binarização de
    acordo com o valor limite passado. 

    ### Params
        img : cv2.Mat
            Imagem openCV (cv2.Mat) de 3 canais
        ?threshold : int = 222
            Valor limite máximo de descarte

    ### Returns
        int
            Média dos pixels da imagem resultado
        tuple[cv2.Mat]
            4 imagens referentes aos diversos passos do processo:
                Imagem original;
                Imagem monocromática;
                Imagem recortada da face;
                Imagem binarizada
    """    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = get_face(gray)
    if face is None: face = gray
    blurred = cv2.GaussianBlur(face, (11, 11), 0)
    thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=8)
    thresh = cv2.dilate(thresh, None, iterations=8)
    return thresh.mean(), (img, gray, face, thresh)
