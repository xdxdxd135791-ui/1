import sys

try:
    import cv2
    import json
    import sqlite3
    import customtkinter as ctk
    from ultralytics import YOLO
    from tkinter import filedialog
    from PIL import Image, ImageTk
    from datetime import datetime
except ModuleNotFoundError as e:
    print(f"Missing dependency: {e.name}")
    print("Install required packages with: pip install -r requirements.txt")
    sys.exit(1)

cfg=json.load(open("config.json"))

conn=sqlite3.connect("people.db")
conn.execute("CREATE TABLE IF NOT EXISTS logs(time TEXT,direction TEXT)")
conn.commit()

model_path=cfg["model"]
model=YOLO(model_path)

cap=cv2.VideoCapture(cfg["camera_id"])

cap.set(cv2.CAP_PROP_FRAME_WIDTH,cfg["width"])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,cfg["height"])

line_x=cfg["line_x"]
conf=cfg["confidence"]

count=0
cam_on=True
mem={}
logs=[]

ctk.set_appearance_mode("dark")

app=ctk.CTk()
app.geometry("1300x750")
app.title("People Counter")

left=ctk.CTkFrame(app,width=250)
left.pack(side="left",fill="y",padx=10,pady=10)

video_label=ctk.CTkLabel(app,text="")
video_label.pack(side="right",expand=True,fill="both")

count_label=ctk.CTkLabel(left,text="COUNT: 0",font=("Arial",24))
count_label.pack(pady=10)

model_label=ctk.CTkLabel(left,text=model_path)
model_label.pack(pady=10)

conf_label=ctk.CTkLabel(left,text=f"CONF: {conf}")
conf_label.pack()

slider=ctk.CTkSlider(left,from_=0.1,to=1.0)
slider.set(conf)
slider.pack(fill="x",padx=20,pady=10)

log_box=ctk.CTkTextbox(left,height=250)
log_box.pack(fill="both",expand=True,padx=10,pady=10)

def toggle_camera():
    global cam_on
    cam_on = not cam_on

ctk.CTkButton(
    left,
    text="Camera On / Off",
    command=toggle_camera
).pack(pady=5)

def load_model():
    global model

    path = filedialog.askopenfilename(
        filetypes=[("YOLO Model", "*.pt")]
    )

    if path:
        model = YOLO(path)
        model_label.configure(text=path)

ctk.CTkButton(
    left,
    text="Load Model",
    command=load_model
).pack(pady=5)

def refresh_logs():
    log_box.delete("1.0", "end")

    for item in logs[:20]:
        log_box.insert("end", item + "\n")

frame_skip = 0

def update():
    global count
    global conf
    global frame_skip

    conf = slider.get()

    conf_label.configure(
        text=f"CONF: {conf:.2f}"
    )

    if cam_on:
        ret, frame = cap.read()

        if ret:
            frame = cv2.resize(
                frame,
                (cfg["width"], cfg["height"])
            )

            frame_skip += 1

            if frame_skip % 3 == 0:
                result = model.track(
                    frame,
                    persist=True,
                    conf=conf,
                    imgsz=320,
                    verbose=False
                )

                cv2.line(
                    frame,
                    (line_x, 0),
                    (line_x, cfg["height"]),
                    (0, 255, 0),
                    2
                )

                if result[0].boxes.id is not None:
                    boxes = result[0].boxes.xyxy.cpu().numpy()
                    ids = result[0].boxes.id.cpu().numpy()

                    for box, track_id in zip(boxes, ids):
                        x1, y1, x2, y2 = map(int, box)
                        cx = (x1 + x2) // 2

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            2
                        )

                        if track_id not in mem:
                            mem[track_id] = cx
                            continue

                        old_x = mem[track_id]
                        mem[track_id] = cx

                        if old_x < line_x and cx >= line_x:
                            count += 1
                            msg = f"{datetime.now()} LEFT_RIGHT"
                            logs.insert(0, msg)

                            conn.execute(
                                "INSERT INTO logs VALUES(?, ?)",
                                (str(datetime.now()), "LEFT_RIGHT")
                            )
                            conn.commit()
                            refresh_logs()

            count_label.configure(
                text=f"COUNT: {count}"
            )

            cv2.putText(
                frame,
                f"COUNT {count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(img)

            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)

    app.after(30, update)

update()
app.mainloop()

cap.release()
conn.close()
cv2.destroyAllWindows()
