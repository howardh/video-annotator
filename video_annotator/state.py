import cv2
import threading
from tqdm import tqdm
import time

class BackgroundTask(threading.Thread):
    def __init__(self,funcs,cleanup=None,progress_callbacks=[],
            done_callback=lambda s: None):
        super().__init__()
        self.funcs = funcs
        self.progress_callbacks = progress_callbacks
        self.done_callback = done_callback
        self.i = None
        self.t = time.process_time()
        self.kill_flag = False
        self.cleanup = cleanup
    def run(self):
        for self.i,func in tqdm(enumerate(self.funcs)):
            func(self.i)
            t = time.process_time()
            if t-self.t > 1:
                for c in self.progress_callbacks:
                    c()
                self.t = t
            if self.kill_flag:
                break
        if self.cleanup is not None:
            self.cleanup()
        self.done_callback(self)
    def kill(self):
        self.kill_flag = True
    def __len__(self):
        return len(self.funcs)

def bounded(v,min_val,max_val):
    return min(max(v,min_val),max_val)

class State:
    def __init__(self, video, annotations):
        self.video = video
        self.annotations = annotations

        self.annotation_id = 0
        self.paused = True
        self.current_frame_index = 0

        self.zoom = 1
        self.zoom_centre = (self.video.width//2, self.video.height//2)

        self.callbacks = {
                'video': [],
                'annotations': [],
                'pause': [],
                'background': []
        }
        self.background_tasks = []

    def add_callback(self, key, callback):
        self.callbacks[key].append(callback)
    def call_callbacks(self, key):
        for f in self.callbacks[key]:
            f()
    def launch_bg_task(self, funcs, cleanup=None):
        def done_callback(task):
            print('done',task)
            self.background_tasks.remove(task)
            if len(self.background_tasks) > 0:
                self.background_tasks[0].start()
            self.call_callbacks('background')
        task = BackgroundTask(funcs,cleanup,self.callbacks['background'],done_callback)
        self.background_tasks.append(task)
        if len(self.background_tasks) == 1:
            task.start()
    def kill_current_bg_task(self):
        self.background_tasks[0].kill()

    def save(self):
        self.annotations.save_annotations()

    def delete_keyframe(self):
        self.annotations.remove_annotation(self.current_frame_index, self.annotation_id)
        self.call_callbacks('video')
        self.call_callbacks('annotations')
    def jump_to_keyframe_nearest(self):
        kf_indices = list(self.annotations[self.annotation_id].keys())
        if len(kf_indices) == 0:
            return
        index = self.current_frame_index
        closest_index = min(kf_indices, key=lambda x: abs(index-x))
        self.current_frame_index = closest_index
        print('%d -> %d' % (index, closest_index))
        self.call_callbacks('video')
        self.call_callbacks('annotations')
    def jump_to_keyframe_prev(self):
        index = self.current_frame_index
        kf_indices = self.annotations[self.annotation_id].manual.keys()
        kf_indices = filter(lambda x: x<index, kf_indices)
        kf_indices = list(kf_indices)
        if len(kf_indices) == 0:
            return
        closest_index = min(kf_indices, key=lambda x: abs(index-x))
        self.current_frame_index = closest_index
        print('%d -> %d' % (index, closest_index))
        self.call_callbacks('video')
        self.call_callbacks('annotations')
    def jump_to_prev_frame(self):
        if self.current_frame_index > 0:
            self.current_frame_index -= 1
            self.call_callbacks('video')
            self.call_callbacks('annotations')
    def jump_to_next_frame(self):
        if self.current_frame_index < self.video.frame_count:
            self.current_frame_index += 1
            self.call_callbacks('video')
            self.call_callbacks('annotations')
    def jump_to_keyframe_next(self):
        index = self.current_frame_index
        kf_indices = self.annotations[self.annotation_id].manual.keys()
        kf_indices = filter(lambda x: x>index, kf_indices)
        kf_indices = list(kf_indices)
        if len(kf_indices) == 0:
            return
        closest_index = min(kf_indices, key=lambda x: abs(index-x))
        self.current_frame_index = closest_index
        print('%d -> %d' % (index, closest_index))
        self.call_callbacks('video')
        self.call_callbacks('annotations')

    def generate_annotations(self):
        funcs = self.annotations[self.annotation_id].template_matched.generate2(
                self.current_frame_index)
        self.launch_bg_task(funcs)
    def generate_annotations_optical_flow(self):
        funcs = self.annotations[self.annotation_id].optical_flow.generate2(
                self.current_frame_index)
        self.launch_bg_task(funcs)
    def generate_annotations_cnn(self):
        funcs = self.annotations.predicted.generate2(
                self.current_frame_index)
        self.launch_bg_task(funcs)
    def generate_annotations_cnn2(self):
        funcs = self.annotations.predicted2.generate2(
                self.current_frame_index)
        self.launch_bg_task(funcs, cleanup=self.annotations.predicted2.map_path)

    def prev_annotation(self):
        ann_ids = sorted(self.annotations.annotations.keys())
        ann_ids.reverse()
        for i in ann_ids:
            if i < self.annotation_id:
                self.annotation_id = i
                print('Selected annotation %d/%d'%(len(ann_ids)-ann_ids.index(self.annotation_id),len(ann_ids)))
                break
        self.call_callbacks('video')
        self.call_callbacks('annotations')
    def next_annotation(self):
        ann_ids = sorted(self.annotations.annotations.keys())
        for i in ann_ids:
            if i > self.annotation_id:
                self.annotation_id = i
                print('Selected annotation %d/%d'%(ann_ids.index(self.annotation_id)+1,len(ann_ids)))
                break
        self.call_callbacks('video')
        self.call_callbacks('annotations')
    def new_annotation(self):
        annotation_id = max(self.annotations.get_ids())+1
        self.annotations[annotation_id] # Access it to create it
        self.annotation_id = annotation_id
        # Console output
        ann_ids = sorted(self.annotations.get_ids())
        print('Selected annotation %d/%d'%(ann_ids.index(self.annotation_id)+1,len(ann_ids)))
        self.call_callbacks('video')
        self.call_callbacks('annotations')
    def delete_annotation(self):
        deleted_id = self.annotation_id
        del self.annotations[deleted_id]
        # Set selected annotation to the previous annotation
        # i.e. Find largest ID that's smaller than the deleted ID
        self.prev_annotation()
        self.next_annotation()
        # Console output
        ann_ids = sorted(self.annotations.get_ids())
        print('Selected annotation %d/%d'%(ann_ids.index(self.annotation_id)+1,len(ann_ids)))
        self.call_callbacks('video')
        self.call_callbacks('annotations')
    def clear_annotation(self):
        self.annotations[self.annotation_id] = {}
        self.call_callbacks('video')
        self.call_callbacks('annotations')

    def inc_window_size(self):
        size = self.annotations[self.annotation_id].get_window_size()[0]
        self.annotations[self.annotation_id].set_window_size(size*2)
        print(size)
        self.call_callbacks('video')
    def dec_window_size(self):
        size = self.annotations[self.annotation_id].get_window_size()[0]
        self.annotations[self.annotation_id].set_window_size(size//2)
        print(size)
        self.call_callbacks('video')
    def inc_template_size(self):
        size = self.annotations[self.annotation_id].get_template_size()[0]
        self.annotations[self.annotation_id].set_template_size(size*2)
        print(size)
        self.call_callbacks('video')
    def dec_template_size(self):
        size = self.annotations[self.annotation_id].get_template_size()[0]
        self.annotations[self.annotation_id].set_template_size(size//2)
        self.call_callbacks('video')

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
        self.call_callbacks('video')
        self.call_callbacks('annotations')

    def toggle_pause(self, p = None):
        if p is None:
            self.paused = not self.paused
        else:
            self.paused = p
        self.call_callbacks('pause')

    def zoom_in(self):
        self.zoom += 0.1
        self.call_callbacks('video')
        print(self.zoom)
    def zoom_out(self):
        self.zoom -= 0.1
        if self.zoom < 1:
            self.zoom = 1
        self.call_callbacks('video')
    def zoom_reset(self):
        self.zoom = 1
        self.call_callbacks('video')
    def zoom_translate(self,dx,dy):
        # Vars
        x,y = self.zoom_centre
        z = self.zoom
        w,h = self.video.width, self.video.height
        # Compute
        x = bounded(x-dx/z,w/z//2,w-w/z//2)
        y = bounded(y-dy/z,h/z//2,h-h/z//2)
        # Save results
        self.zoom_centre = x,y
        print('translate',dx,dy)
        self.call_callbacks('video')

    def get_frame(self, render_annotations=True):
        frame = self.video.get_frame(self.current_frame_index)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        if render_annotations:
            frame = self.annotations.render(frame, self.current_frame_index)
        # Compute Zoom Size
        h,w,_ = frame.shape
        zx,zy = self.zoom_centre
        zx = int(zx)
        zy = int(zy)
        zw = int(w//self.zoom)
        zh = int(h//self.zoom)
        left = bounded(zx-zw//2,0,w-zw)
        top = bounded(zy-zh//2,0,h-zh)
        cropped_frame = frame[top:top+zh,left:left+zw,:]
        frame = cv2.resize(cropped_frame,(w,h))
        return frame
    
    def save_video(self, file_name, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = self.current_frame_index
        if end_frame is None:
            end_frame = self.video.frame_count
        print('Saving video to',file_name)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output = cv2.VideoWriter(file_name, fourcc, self.video.fps, (self.video.width,self.video.height))
        def save_frame(index):
            frame = self.video.get_frame(index)
            frame = self.annotations.render(frame, index)
            output.write(frame)
        def cleanup():
            output.release()
            print('output file released')
        self.launch_bg_task([lambda i: save_frame(i+start_frame) for _ in range(end_frame-start_frame)],cleanup=cleanup)

    def add_annotation(self,annotation):
        if annotation is not None:
            z = self.zoom
            zx,zy = self.zoom_centre
            w,h = self.video.width, self.video.height
            x,y = annotation
            x = (zx-w/z/2+x*w/z)/w
            y = (zy-h/z/2+y*h/z)/h
            annotation = x,y
        self.annotations.add_annotation(
                frame_index=self.current_frame_index,
                annotation_id=self.annotation_id,
                annotation=annotation)
        self.call_callbacks('video')
        self.call_callbacks('annotations')

    def update(self):
        if self.paused:
            return
        if self.current_frame_index >= self.video.frame_count-1:
            self.paused = True
            return
        self.jump_to_next_frame()
