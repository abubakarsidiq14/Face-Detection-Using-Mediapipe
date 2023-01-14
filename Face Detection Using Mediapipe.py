import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_utils = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles

cam = cv2.VideoCapture(0)

while cam.isOpened():
    rt,img=cam.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = mp_face_mesh.FaceMesh(refine_landmarks=True).process(img)

    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:

            mp_utils.draw_landmarks(
                image = img,
                landmark_list = face_landmarks,
                connections = mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec = None,
                connection_drawing_spec = mp_drawing_style.get_default_face_mesh_contours_style(),
            )

            '''mp_utils.draw_landmarks(
                image = img,
                landmark_list = face_landmarks,
                connections = mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec = None,
                connection_drawing_spec = mp_drawing_style.get_default_face_mesh_tesselation_style(),
            )'''

    cv2.imshow("Frame",img)
    cv2.waitKey(20)
    if  0xFF == ord('q'):
        cv2.destroyAllWindows()



