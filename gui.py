import tkinter
import cv2
import PIL
import PIL.Image, PIL.ImageTk
import numpy as np
import time

import templatematcher

class App:
    SEEKBAR_H_PADDING=10
    SEEKBAR_HEIGHT=40

    def __init__(self, window, video):
        self.window = window
        self.window.title('Video Annotator')
        self.video = video

        self.paused = True
        self.current_frame_index = 0
        self.annotation_id = 0
        self.last_update = time.process_time()

        #self.canvas = tkinter.Canvas(
        #        window, width=video.width, height=video.height)
        self.canvas = tkinter.Canvas(window)
        self.seekbar = tkinter.Canvas(
                window, width=video.width, height=App.SEEKBAR_HEIGHT)

        self.canvas.pack()
        self.seekbar.pack()
        self.create_menu()

        self.window.bind('<Configure>', self.handle_resize)
        self.window.bind('q', self.quit)
        self.window.bind('<space>', self.toggle_pause)
        self.window.bind('n', self.jump_to_keyframe_nearest)
        self.window.bind('<Control-Left>', self.jump_to_keyframe_prev)
        self.window.bind('<Control-Right>', self.jump_to_keyframe_next)
        self.window.bind('<Left>', self.jump_to_prev_frame)
        self.window.bind('<Right>', self.jump_to_next_frame)
        self.window.bind('<Up>', lambda e: self.prev_annotation())
        self.window.bind('<Down>', lambda e: self.next_annotation())
        self.window.bind('<Delete>', self.delete_keyframe)

        self.canvas.bind('<Button-1>', self.handle_video_click)
        self.canvas.bind('<Button-3>', self.handle_video_click)
        self.seekbar.bind('<Button-1>', self.handle_seekbar_click)

        self.update()
        self.window.mainloop()

    def create_menu(self):
        menu_bar = tkinter.Menu(self.window)

        file_menu = tkinter.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Open", command=lambda: None)
        file_menu.add_command(label="Save", command=lambda: None)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)

        edit_menu = tkinter.Menu(menu_bar, tearoff=0)
        edit_menu.add_command(label="New Annotation", command=self.new_annotation)
        edit_menu.add_command(label="Delete Annotation", command=self.delete_annotation)
        edit_menu.add_separator()
        edit_menu.add_command(label="Clear Annotation",
                command=self.clear_annotation)
        edit_menu.add_separator()
        edit_menu.add_command(label="Generate Annotation Path",
                command=self.generate_annotations)
        menu_bar.add_cascade(label="Edit", menu=edit_menu)

        self.window.config(menu=menu_bar)

    def handle_resize(self, event):
        w_width = self.window.winfo_width()
        w_height = self.window.winfo_height()
        v_width = self.video.width
        v_height = self.video.height
        scale = min(w_width/v_width,w_height/v_height)

        if w_height > 100:
            if scale*v_height > w_height-100:
                scale = (w_height-100)/v_height

        # Compute new canvas size
        canvas_width = scale*v_width
        canvas_height = scale*v_height

        self.canvas.config(width=canvas_width,height=canvas_height)
        self.render_current_frame()
        self.render_seekbar()

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

    def jump_to_prev_frame(self, event):
        if self.current_frame_index > 0:
            self.current_frame_index -= 1
            self.render_current_frame()

    def jump_to_next_frame(self, event):
        if self.current_frame_index < self.video.frame_count:
            self.current_frame_index += 1
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

    def generate_annotations(self):
        self.video.generate_annotations(self.annotation_id)
        self.render_current_frame()
        self.render_seekbar()

    def prev_annotation(self):
        ann_ids = sorted(self.video.annotations.keys())
        ann_ids.reverse()
        print(ann_ids)
        for i in ann_ids:
            if i < self.annotation_id:
                self.annotation_id = i
                print('Selected annotation',self.annotation_id)
                break
        self.render_seekbar()

    def next_annotation(self):
        ann_ids = sorted(self.video.annotations.keys())
        print(ann_ids)
        for i in ann_ids:
            if i > self.annotation_id:
                self.annotation_id = i
                print('Selected annotation',self.annotation_id)
                break
        self.render_seekbar()

    def new_annotation(self):
        annotation_id = max(self.video.annotations.keys())+1
        self.video.annotations[annotation_id] = {}
        self.annotation_id = annotation_id
        self.render_seekbar()

    def delete_annotation(self):
        deleted_id = self.annotation_id
        del self.video.annotations[deleted_id]
        # Set selected annotation to the previous annotation
        # i.e. Find largest ID that's smaller than the deleted ID
        self.prev_annotation()
        self.next_annotation()

    def clear_annotation(self):
        self.video.annotations[self.annotation_id] = {}
        self.render_seekbar()

    def seek(self, frame):
        if type(frame) is int:
            self.current_frame_index = frame
        elif type(frame) is float:
            self.current_frame_index = int(frame*self.video.frame_count)
        else:
            raise TypeError('Unsupported type: %s. Expected int or float.' % (type(frame)))
        if self.current_frame_index < 0:
            self.current_frame_index = 0
        if self.current_frame_index > self.video.frame_count:
            self.current_frame_index = self.video.frame_count
        self.render_current_frame()

    def handle_video_click(self, event):
        if event.num == 1:
            width = event.widget.winfo_width()
            height = event.widget.winfo_height()
            self.video.add_annotation(frame_index=self.current_frame_index,
                    annotation_id=self.annotation_id,
                    annotation=(event.x/width, event.y/height))
        elif event.num == 3:
            self.video.add_annotation(frame_index=self.current_frame_index,
                    annotation_id=self.annotation_id,
                    annotation=None)
        self.render_current_frame()
        self.render_seekbar()

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
        c_width = self.canvas.winfo_width()
        c_height = self.canvas.winfo_height()
        v_width = self.video.width
        v_height = self.video.height
        scale = min(c_width/v_width,c_height/v_height)
        dims = (int(v_width*scale),int(v_height*scale))
        if dims[0] < 1 or dims[1] < 1:
            return
        frame = cv2.resize(frame, dims)
        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
        self.canvas.create_image(c_width/2, c_height/2, image=self.photo, anchor=tkinter.CENTER)

    def render_seekbar(self):
        width = self.seekbar.winfo_width()
        height = self.seekbar.winfo_height()
        h_padding = App.SEEKBAR_H_PADDING

        # Background
        self.seekbar.create_rectangle(0, 0, width, height, fill='white')

        # Annotation markers
        def draw_annotations(colour1, colour2, anns):
            for frame,ann in anns.items():
                if ann is None:
                    colour = colour2
                else:
                    colour = colour1
                pos = frame/self.video.frame_count*(width-h_padding*2)
                self.seekbar.create_line(
                        h_padding+pos, 0, h_padding+pos, height, fill=colour)
        for ann_id,anns in self.video.annotations.items():
            if ann_id == self.annotation_id:
                continue
            draw_annotations('#cccccc','#ffcccc',anns)
        draw_annotations('black','red',self.video.annotations[self.annotation_id])

        # Current position marker
        pos = self.current_frame_index/self.video.frame_count*(width-h_padding*2)
        polygon = np.array([[0,0],[5,-10],[-5,-10]], dtype=np.float)
        polygon += [h_padding+pos,height/2]
        polygon = polygon.flatten().tolist()
        self.seekbar.create_polygon(polygon, fill='black')
        self.seekbar.create_line(
                h_padding, height/2, width-h_padding, height/2)

    def update(self):
        if self.paused:
            self.render_current_frame()
            self.render_seekbar()
            return

        self.current_frame_index += 1 # FIXME: This skips the first frame
        self.render_current_frame()
        self.render_seekbar()

        new_process_time = time.process_time()
        time_diff = new_process_time-self.last_update
        self.last_update = new_process_time
        #delay = int(1000/self.video.fps-time_diff)
        delay = 1
        self.window.after(delay, self.update)
