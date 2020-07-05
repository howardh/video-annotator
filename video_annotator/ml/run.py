import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='foo')

    parser_train = subparsers.add_parser('train', help='train help')
    parser_train.set_defaults(which='train')
    parser_train.add_argument('--checkpoint_dir', type=str, default='./Checkpoint',
            help='Directory in which checkpoint files are saved. The directory will not be created, so it must exist.')
    parser_train.add_argument('--model', type=str, default='',
            help='')

    parser_prior = subparsers.add_parser('prior', help='prior help')
    parser_prior.set_defaults(which='prior')

    parser_dataset = subparsers.add_parser('dataset', help='dataset help')
    parser_dataset.set_defaults(which='dataset')
    parser_dataset.add_argument('--dataset_dir', type=str, default='./dataset',
            help='')

    parser_annotate = subparsers.add_parser('annotate', help='dataset help')
    parser_annotate.set_defaults(which='annotate')
    parser_annotate.add_argument('--checkpoint', type=str, required=True,
            help='Path to the model checkpoint to load.')
    parser_annotate.add_argument('--input', type=str, required=True,
            help='Path to the video to annotate.')
    parser_annotate.add_argument('--output', type=str, required=True,
            help='Path of the output annotated video.')

    args = parser.parse_args()

    if args.which == 'dataset':
        from video_annotator.ml.dataset import VideoDataset
        dataset = VideoDataset(args.dataset_dir)
        dataset.to_photo_dataset()
    elif args.which == 'train':
        video_annotator.ml.dataset.train_anchor_box()
    elif args.which == 'prior':
        pass
    elif args.which == 'annotate':
        import video_annotator
        import video_annotator.ml
        import video_annotator.annotation
        from video_annotator.video import Video
        from tqdm import tqdm
        import cv2

        input_file_path = args.input
        output_file_path = args.output

        # Load Video
        video = Video(input_file_path)

        # Annotate video
        annotation = video_annotator.annotation.PredictedAnnotations2(video)
        for frame_index in tqdm(range(video.frame_count), desc='annotating'):
            annotation[frame_index] = annotation.search_frame(frame_index)
        annotation.postprocess()

        # Save video with annotation
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output = cv2.VideoWriter(output_file_path, fourcc, video.fps, (video.width,video.height))
        for frame_index in tqdm(range(video.frame_count), desc='saving video'):
            frame = video.get_frame(frame_index)
            annotation.render(frame, frame_index, colour=(255,255,255))
            output.write(frame)

        output.release()
        video.close()
