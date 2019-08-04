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
        self.window.bind('n', self.jump_to_keyframe_nearest)
        self.window.bind('<Left>', self.jump_to_keyframe_prev)
        self.window.bind('<Right>', self.jump_to_keyframe_next)
        self.window.bind('<Delete>', self.delete_keyframe)

        self.update()
        self.window.mainloop()

    def delete_keyframe(self, event):
        self.video.remove_annotation(self.current_frame_index, self.annotation_id)
        self.render_current_frame()

    def jump_to_keyframe_nearest(self, event):
        kf_indices = list(self.video.annotations[self.annotation_id].keys())
        if len(kf_indices) == 0:
            return
        index = self.current_frame_index
        closest_index = min(kf_indices, key=lambda x: abs(index-x))
        self.current_frame_index = closest_index
        print('%d -> %d' % (index, closest_index))
        self.render_current_frame()

    def jump_to_keyframe_prev(self, event):
        index = self.current_frame_index
        kf_indices = self.video.annotations[self.annotation_id].keys()
        kf_indices = filter(lambda x: x<index, kf_indices)
        kf_indices = list(kf_indices)
        if len(kf_indices) == 0:
            return
        closest_index = min(kf_indices, key=lambda x: abs(index-x))
        self.current_frame_index = closest_index
        print('%d -> %d' % (index, closest_index))
        self.render_current_frame()

    def jump_to_keyframe_next(self, event):
        index = self.current_frame_index
        kf_indices = self.video.annotations[self.annotation_id].keys()
        kf_indices = filter(lambda x: x>index, kf_indices)
        kf_indices = list(kf_indices)
        if len(kf_indices) == 0:
            return
        closest_index = min(kf_indices, key=lambda x: abs(index-x))
        self.current_frame_index = closest_index
        print('%d -> %d' % (index, closest_index))
        self.render_current_frame()

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
