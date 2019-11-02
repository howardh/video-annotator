import tkinter
import cv2
import PIL
import PIL.Image, PIL.ImageTk
import numpy as np

class App:
    SEEKBAR_H_PADDING=10
    def __init__(self, window, video):
        self.window = window
        self.window.title('Video Annotator')
        self.video = video

        self.paused = False
        self.current_frame_index = 0
        self.annotation_id = 0

        self.canvas = tkinter.Canvas(window, width=video.width, height=video.height)
        self.seekbar = tkinter.Canvas(window, width=video.width, height=40)
        self.canvas.pack()
        self.seekbar.pack()

        self.window.bind('<Configure>', self.handle_resize)
        self.window.bind('q', self.quit)
        self.window.bind('<space>', self.toggle_pause)
        self.window.bind('n', self.jump_to_keyframe_nearest)
        self.window.bind('<Left>', self.jump_to_keyframe_prev)
        self.window.bind('<Right>', self.jump_to_keyframe_next)
        self.window.bind('<Delete>', self.delete_keyframe)

        self.canvas.bind('<Button-1>', self.handle_video_click)
        self.seekbar.bind('<Button-1>', self.handle_seekbar_click)

        self.update()
        self.window.mainloop()

    def handle_resize(self, event):
        window_width = event.width
        window_height = event.height
        canvas_width = window_width
        canvas_height = self.video.height*(window_width/self.video.width)
        self.canvas.config(width=canvas_width,height=canvas_height)

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

    def seek(self, frame):
        if type(frame) is int:
            self.current_frame_index = frame
        elif type(frame) is float:
            self.current_frame_index = int(frame*self.video.frame_count)
        else:
            raise TypeError('Unsupported type: %s. Expected int or float.' % (type(frame)))
        self.render_current_frame()

    def handle_video_click(self, event):
        width = event.widget.winfo_width()
        height = event.widget.winfo_height()
        self.video.add_annotation(frame_index=self.current_frame_index,
                annotation_id=self.annotation_id,
                annotation=(event.x/width, event.y/height))
        self.render_current_frame()

    def handle_seekbar_click(self, event):
        h_padding = App.SEEKBAR_H_PADDING
        width = event.widget.winfo_width()
        height = event.widget.winfo_height()
        seekto_percent = (event.x-h_padding)/(width-h_padding*2)
        print('Seekbar click',width,height,event.x,event.y)
        self.seek(seekto_percent)
        self.render_seekbar()

    def toggle_pause(self, event):
        self.paused = not self.paused
        if not self.paused:
            self.update()

    def quit(self, event):
        self.window.destroy()

    def render_current_frame(self):
        frame = self.video.get_frame(self.current_frame_index, True)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.canvas.winfo_width(), self.canvas.winfo_height()))
        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

    def render_seekbar(self):
        width = self.seekbar.winfo_width()
        height = self.seekbar.winfo_height()
        h_padding = App.SEEKBAR_H_PADDING

        # Background
        self.seekbar.create_rectangle(0, 0, width, height, fill='white')

        # Annotation markers
        for ids,anns in self.video.annotations.items():
            for frame,_ in anns.items():
                pos = frame/self.video.frame_count*(width-h_padding*2)
                self.seekbar.create_line(h_padding+pos, 0, h_padding+pos, height)

        # Current position marker
        pos = self.current_frame_index/self.video.frame_count*(width-h_padding*2)
        polygon = np.array([[0,0],[5,-10],[-5,-10]], dtype=np.float)
        polygon += [h_padding+pos,height/2]
        polygon = polygon.flatten().tolist()
        self.seekbar.create_polygon(polygon, fill='black')
        self.seekbar.create_line(h_padding, height/2, width-h_padding, height/2)

    def update(self):
        if self.paused:
            return
        self.current_frame_index += 1 # FIXME: This skips the first frame
        self.render_current_frame()
        self.render_seekbar()

        delay = int(1000/self.video.fps)-15
        self.window.after(delay, self.update)
