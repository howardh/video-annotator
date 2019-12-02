import tkinter
import cv2
import PIL
import PIL.Image, PIL.ImageTk
import numpy as np
import time

import templatematcher

class App:
    def __init__(self, window, state):
        self.window = window
        self.window.title('Video Annotator')
        self.state = state

        self.last_update = time.process_time()

        self.canvas = tkinter.Canvas(window)
        self.canvas.pack()
        self.seekbar = SeekBar(window, state)

        self.image_id = None

        self.create_menu()

        self.window.bind('<Configure>', self.handle_resize)

        self.window.bind('<Control-s>', lambda e: self.state.save())
        self.window.bind('q', lambda e: self.quit())
        self.window.bind('<space>', lambda e: self.toggle_pause())
        self.window.bind('n', lambda e: self.state.jump_to_keyframe_nearest())
        self.window.bind('<Control-Left>',
                lambda e: self.state.jump_to_keyframe_prev())
        self.window.bind('<Control-Right>',
                lambda e: self.state.jump_to_keyframe_next())
        self.window.bind('<Left>', lambda e: self.state.jump_to_prev_frame())
        self.window.bind('<Right>', lambda e: self.state.jump_to_next_frame())
        self.window.bind('<Up>', lambda e: self.state.prev_annotation())
        self.window.bind('<Down>', lambda e: self.state.next_annotation())
        self.window.bind('<Delete>', lambda e: self.state.delete_keyframe())

        self.canvas.bind('<Button-1>', self.handle_video_click)
        self.canvas.bind('<Button-3>', self.handle_video_click)
        self.window.bind('g', lambda e: self.generate_annotations())

        self.state.add_callback('video',self.render_current_frame)
        self.state.add_callback('annotations',self.seekbar.render)
        self.state.add_callback('pause',self.update)

        self.update()
        self.window.mainloop()

    def create_menu(self):
        menu_bar = tkinter.Menu(self.window)

        file_menu = tkinter.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Open", command=lambda: None)
        file_menu.add_command(label="Save", command=self.state.save)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)

        edit_menu = tkinter.Menu(menu_bar, tearoff=0)
        edit_menu.add_command(label="New Annotation", command=self.state.new_annotation)
        edit_menu.add_command(label="Delete Annotation", command=self.state.delete_annotation)
        edit_menu.add_separator()
        edit_menu.add_command(label="Clear Annotation",
                command=self.state.clear_annotation)
        edit_menu.add_separator()
        edit_menu.add_command(label="Generate Annotation Path",
                command=self.state.generate_annotations)
        menu_bar.add_cascade(label="Edit", menu=edit_menu)

        self.window.config(menu=menu_bar)

    def handle_resize(self, event):
        w_width = self.window.winfo_width()
        w_height = self.window.winfo_height()
        v_width = self.state.video.width
        v_height = self.state.video.height
        scale = min(w_width/v_width,w_height/v_height)

        if w_height > 100:
            if scale*v_height > w_height-100:
                scale = (w_height-100)/v_height

        # Compute new canvas size
        canvas_width = scale*v_width
        canvas_height = scale*v_height

        self.canvas.config(width=canvas_width,height=canvas_height)
        self.seekbar.resize(w_width)
        print('resize',canvas_width,canvas_height)
        self.render_current_frame()
        self.seekbar.render()

    def handle_video_click(self, event):
        if event.num == 1:
            width = event.widget.winfo_width()
            height = event.widget.winfo_height()
            self.state.add_annotation(
                    annotation=(event.x/width, event.y/height))
        elif event.num == 3:
            self.state.add_annotation(annotation=None)

    def toggle_pause(self):
        self.state.toggle_pause()

    def quit(self):
        self.window.destroy()

    def render_current_frame(self):
        frame = self.state.get_frame(render_annotations=True)
        c_width = float(self.canvas.cget('width'))
        c_height = float(self.canvas.cget('height'))
        v_height, v_width, _ = frame.shape
        scale = min(c_width/v_width,c_height/v_height)
        dims = (int(v_width*scale),int(v_height*scale))
        if dims[0] < 1 or dims[1] < 1:
            return
        frame = cv2.resize(frame, dims)
        self.canvas.delete('all')
        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
        self.image_id = self.canvas.create_image(c_width/2, c_height/2, image=self.photo, anchor=tkinter.CENTER)

    def update(self):
        self.state.update()
        if self.state.paused:
            return

        new_process_time = time.process_time()
        time_diff = new_process_time-self.last_update
        self.last_update = new_process_time
        #delay = int(1000/self.video.fps-time_diff)
        delay = 1
        self.window.after(delay, self.update)

class SeekBar:
    SEEKBAR_H_PADDING=10
    SEEKBAR_HEIGHT=40

    def __init__(self, parent, state):
        self.parent = parent
        self.state = state
        self.canvas = tkinter.Canvas(self.parent,
                width=self.parent.winfo_width(),
                height=self.SEEKBAR_HEIGHT)
        print(self.parent.winfo_width())
        self.canvas.pack()

        self.canvas.bind('<Button-1>', self.handle_click)

    def resize(self, width):
        self.canvas.config(width=width,height=self.SEEKBAR_HEIGHT)
        self.render()

    def handle_click(self, event):
        h_padding = self.SEEKBAR_H_PADDING
        width = event.widget.winfo_width()
        height = event.widget.winfo_height()
        seekto_percent = (event.x-h_padding)/(width-h_padding*2)
        print('Seekbar click',width,height,event.x,event.y)
        self.state.seek(seekto_percent)
        self.render()

    def render(self):
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        h_padding = SeekBar.SEEKBAR_H_PADDING

        self.canvas.delete('all')

        # Background
        self.canvas.create_rectangle(0, 0, width, height, fill='white')

        # Annotation markers
        def draw_annotations(colour1, colour2, anns):
            for frame,ann in anns.items():
                if ann is None:
                    colour = colour2
                else:
                    colour = colour1
                pos = frame/self.state.video.frame_count*(width-h_padding*2)
                self.canvas.create_line(
                        h_padding+pos, 0, h_padding+pos, height, fill=colour)
        for ann_id,anns in self.state.annotations.annotations.items():
            if ann_id == self.state.annotation_id:
                continue
            draw_annotations('#cccccc','#ffcccc',anns.manual)
        draw_annotations('black','red',self.state.annotations[self.state.annotation_id].manual)

        # Current position marker
        pos = self.state.current_frame_index/self.state.video.frame_count*(width-h_padding*2)
        polygon = np.array([[0,0],[5,-10],[-5,-10]], dtype=np.float)
        polygon += [h_padding+pos,height/2]
        polygon = polygon.flatten().tolist()
        self.canvas.create_polygon(polygon, fill='black')
        self.canvas.create_line(
                h_padding, height/2, width-h_padding, height/2)
