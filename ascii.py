import cv2
import mediapipe as mp
import numpy as np
import random


mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands


WIDTH, HEIGHT = 640, 480


ORANGE = (0, 165, 255)  
PURPLE = (255, 0, 255)
WHITE = (255, 255, 255)
RED = (0, 0, 255)
GOLD = (0, 215, 255)
GRAY = (200, 200, 200)


hand_smoke_particles = []
mouth_smoke_particles = []

lighter_active = False


def draw_cigarette(frame, x1, y1, x2, y2, lit=False):
    cv2.line(frame, (x1, y1), (x2, y2), WHITE, 8)  
    tip_x = int((x1 + x2) / 2)
    tip_y = int((y1 + y2) / 2)

  
    if lit:
        cv2.circle(frame, (tip_x, tip_y), 6, RED, -1)
        cv2.circle(frame, (tip_x, tip_y), 10, ORANGE, 2)
    return tip_x, tip_y

def draw_lighter(frame, x, y, active=False):
    if active:
        cv2.rectangle(frame, (x - 8, y - 25), (x + 8, y), GOLD, -1)
        cv2.circle(frame, (x, y - 30), 12, GOLD, -1)


def is_mouth_open(face_landmarks, w, h):
    top_lip = face_landmarks.landmark[13]
    bottom_lip = face_landmarks.landmark[14]
    dist = abs(top_lip.y - bottom_lip.y) * h
    return dist > 18

def create_smoke(particles, origin, count=3, radius=(5, 15), alpha=1.0):
    for _ in range(count):
        particles.append([
            origin[0],
            origin[1],
            random.randint(radius[0], radius[1]),
            alpha
        ])


def draw_smoke(frame, particles):
    new_particles = []
    for (x, y, r, alpha) in particles:
        overlay = frame.copy()
        cv2.circle(overlay, (x, y), r, GRAY, -1)
        frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)

      
        y -= 2
        alpha -= 0.02
        if alpha > 0:
            new_particles.append([x + random.randint(-1, 1), y, r, alpha])
    return frame, new_particles


cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1) as face_mesh, \
     mp_hands.Hands(max_num_hands=2) as hands:  

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        #
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        
        face_results = face_mesh.process(rgb)
        hand_results = hands.process(rgb)

       
        display_frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        mouth_center = None
        exhaling = False
        cig_tip = None
        lighter_pos = None
        lit_cig = False

      
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    x, y = int(lm.x * WIDTH), int(lm.y * HEIGHT)
                    cv2.circle(display_frame, (x, y), 2, ORANGE, -1)

           
                top_lip = face_landmarks.landmark[13]
                bottom_lip = face_landmarks.landmark[14]
                mx = int((top_lip.x + bottom_lip.x) / 2 * WIDTH)
                my = int((top_lip.y + bottom_lip.y) / 2 * HEIGHT)
                mouth_center = (mx, my)

               
                if is_mouth_open(face_landmarks, WIDTH, HEIGHT):
                    exhaling = True

        
        if hand_results.multi_hand_landmarks:
            for hand_index, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                landmarks = []
                for idx, lm in enumerate(hand_landmarks.landmark):
                    x, y = int(lm.x * WIDTH), int(lm.y * HEIGHT)
                    landmarks.append((x, y))

                   
                    cv2.circle(display_frame, (x, y), 5, PURPLE, -1)
                    cv2.putText(display_frame, str(idx), (x + 5, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

                
                for connection in mp_hands.HAND_CONNECTIONS:
                    start_idx, end_idx = connection
                    x1, y1 = landmarks[start_idx]
                    x2, y2 = landmarks[end_idx]
                    cv2.line(display_frame, (x1, y1), (x2, y2), PURPLE, 2)

                
                if hand_index == 0 and len(landmarks) > 8:
                    cig_tip = draw_cigarette(display_frame, landmarks[4][0], landmarks[4][1],
                                             landmarks[8][0], landmarks[8][1], lit=lighter_active)

                    
                    if lighter_active and mouth_center and abs(landmarks[8][0] - mouth_center[0]) < 60 and abs(landmarks[8][1] - mouth_center[1]) < 60:
                        create_smoke(hand_smoke_particles, cig_tip, count=2)

                
                if hand_index == 1 and len(landmarks) > 8:
                    lighter_pos = landmarks[8]  # Index finger tip
                    draw_lighter(display_frame, lighter_pos[0], lighter_pos[1], active=lighter_active)

                    
                    if cig_tip and abs(lighter_pos[0] - cig_tip[0]) < 40 and abs(lighter_pos[1] - cig_tip[1]) < 40:
                        lighter_active = True
                    else:
                        lighter_active = False

        
        if exhaling and mouth_center:
            create_smoke(mouth_smoke_particles, mouth_center, count=5, radius=(10, 20))

       
        display_frame, hand_smoke_particles = draw_smoke(display_frame, hand_smoke_particles)
        display_frame, mouth_smoke_particles = draw_smoke(display_frame, mouth_smoke_particles)

        # Show final result
        cv2.imshow("Cigarette + Lighter Simulation", display_frame)

        
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
