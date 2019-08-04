import tkinter
import cv2
import PIL
import PIL.Image, PIL.ImageTk

class App:
    def __init__(self, window, video):
        self.window = window
        self.window.title('Video Annotator')
        self.video = video

        self.paused = False
        self.current_frame_index = 0
        self.annotation_id = 0

        self.canvas = tkinter.Canvas(window, width=self.video.width, height=self.video.height)
        self.canvas.pack()
        self.window.bind('q', self.quit)
        self.window.bind('<space>', self.toggle_pause)
        self.window.bind('<Button-1>', self.handle_click)

        self.update()
        self.window.mainloop()

    def handle_click(self, event):
        self.video.add_annotation(frame_index=self.current_frame_index,
                annotation_id=self.annotation_id,
                annotation=(event.x, event.y))
        self.render_current_frame()

    def toggle_pause(self, event):
        self.paused = not self.paused
        if not self.paused:
            self.update()

    def quit(self, event):
        self.window.destroy()

    def render_current_frame(self):
        frame = self.video.get_frame(self.current_frame_index, True)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

    def update(self):
        if self.paused:
            return
        self.current_frame_index += 1 # FIXME: This skips the first frame
        self.render_current_frame()

        delay = int(1000/self.video.fps)-5
        self.window.after(delay, self.update)
