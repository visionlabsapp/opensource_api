from fastapi import FastAPI

import math
import cv2
import mediapipe as mp

app = FastAPI()

# coordinates for face parts
FACE_OUTLINE = [10, 454, 152, 234]
MOUTH_OUTLINE = [0, 291, 17, 61]
LEFT_EYE_OUTLINE = [386, 263, 374, 362]
RIGHT_EYE_OUTLINE = [159, 133, 145, 33]

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/face_ratio/{imagepath}")
def face_ratio(imagepath):
    
    return [{"face_ratio": "0.223", "mouth_ratio": "0.456", "left_eye_ratio": "0.234", "right_eye_ratio": "0.24",}]

@app.get("/personalitytype/{imagepath}")
def personalitytype(imagepath):
    type = ""
    #out_folder = f"{self.working_dir}/{id}/keyframes"
    #img_path = f"{out_folder}\\{q}\\{t}.jpg"
    
    #add image path and display in the html5
    #img_path = "C:/Users/user/tavy_fastapi/v0.1/images/Michelle_Chan.jpg"
    img_path = "C:/Users/user/visionlabs/v1.0/cvapi_deploy/images/"+imagepath
    
    out = imageanalysis(imagepath, False);
    
    return [{"face_ratio": out['face_ratio'], "mouth_ratio": out['mouth_ratio'], "left_eye_ratio": out['left_eye_ratio'], "right_eye_ratio": out['right_eye_ratio']}]

def get_single_face_mesh(img_obj):
    """get a face mesh in a given image, if there is more than one face,
    always return the first face in the face detection result from mediapipe
    
    param img_obj: image object read from cv2

    Return: a face mesh
    """
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img_obj, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks: return None
        return results.multi_face_landmarks[0]        
        
def imageanalysis(image, debug=True):
    img_path = "C:/Users/user/visionlabs/v1.0/cvapi_deploy/images/"+image
    
    # mediapipe
    img_obj = cv2.imread(img_path)
    #img_obj = crop_img(img_obj)
    face_landmark = get_single_face_mesh(img_obj).landmark
    print(f"after face crop: {img_obj.shape}")
                    
    dt = {
        "video": id
        #, "img_file": f"{time}.jpg"
        , "img_file": " 123.jpg"
        , "face_ratio": landmark_area_ratio(face_landmark, FACE_OUTLINE)
        , "mouth_ratio": landmark_area_ratio(face_landmark, MOUTH_OUTLINE)
        , "left_eye_ratio": landmark_area_ratio(face_landmark, LEFT_EYE_OUTLINE)
        , "right_eye_ratio": landmark_area_ratio(face_landmark, RIGHT_EYE_OUTLINE)
    }

    return dt

def distance(p1, p2):
    """3D distance between two points
    
    param p1, p2: points with 3D coordinates

    Return: distance between p1 and p2
    """    
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p1.z)**2)
    
def landmark_area_ratio(face_landmark, mask):
    """calculate the width/hieght ratio of a 3D square area in a face landmark
    
    param face_landmark: the face landmark object
    mask: the 4 points of the 3D square area, starts from right, clockwise

    Return: the width/hieght ratio of the 3D square area
    """
    
    top = face_landmark[mask[0]]
    right = face_landmark[mask[1]]
    bottom = face_landmark[mask[2]]
    left = face_landmark[mask[3]]
    return distance(right, left) / distance(top, bottom)                
