import cv2
import threading
from tqdm import tqdm
import time

class BackgroundTask(threading.Thread):
    def __init__(self,funcs,progress_callbacks=[],
            done_callback=lambda s: None):
        super().__init__()
        self.funcs = funcs
        self.progress_callbacks = progress_callbacks
        self.done_callback = done_callback
        self.i = None
        self.t = time.process_time()
    def run(self):
        for self.i,func in tqdm(enumerate(self.funcs)):
            func(self.i)
            t = time.process_time()
            if t-self.t > 1:
                for c in self.progress_callbacks:
                    c()
                self.t = t
        self.done_callback(self)
    def __len__(self):
        return len(self.funcs)

class State:
    def __init__(self, video, annotations):
        self.video = video
        self.annotations = annotations

        self.annotation_id = 0
        self.paused = True
        self.current_frame_index = 0

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

    def launch_bg_task(self, funcs):
        def done_callback(task):
            print('done',task)
            self.background_tasks.remove(task)
            if len(self.background_tasks) > 0:
                self.background_tasks[0].start()
            self.call_callbacks('background')
        task = BackgroundTask(funcs,self.callbacks['background'],done_callback)
        self.background_tasks.append(task)
        if len(self.background_tasks) == 1:
            task.start()

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
        self.render_seekbar()
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

    def get_frame(self, render_annotations=True):
        frame = self.video.get_frame(self.current_frame_index)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        if render_annotations:
            frame = self.annotations.render(frame, self.current_frame_index)
        return frame
    
    def add_annotation(self,annotation):
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


