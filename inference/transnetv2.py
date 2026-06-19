import os
import numpy as np
import tensorflow as tf


class TransNetV2:

    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), "transnetv2-weights/")
            if not os.path.isdir(model_dir):
                raise FileNotFoundError(f"[TransNetV2] ERROR: {model_dir} is not a directory.")
            else:
                print(f"[TransNetV2] Using weights from {model_dir}.")

        self._input_size = (27, 48, 3)

        try:
            self._model = tf.saved_model.load(model_dir)
        except OSError as exc:
            raise IOError(
                f"[TransNetV2] It seems that files in {model_dir} are corrupted or missing."
            ) from exc

        self._predict_fn = self._build_predict_fn()

    def _build_predict_fn(self):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=[None, 100, 27, 48, 3], dtype=tf.uint8)
            ]
        )
        def predict_fn(frames):
            frames = tf.cast(frames, tf.float32)
            logits, dict_ = self._model(frames)

            single_frame_pred = tf.sigmoid(logits)
            all_frames_pred = tf.sigmoid(dict_["many_hot"])

            return single_frame_pred, all_frames_pred

        return predict_fn

    def predict_raw(self, frames: np.ndarray):
        assert len(frames.shape) == 5 and frames.shape[2:] == self._input_size, \
            "[TransNetV2] Input shape must be [batch, frames, height, width, 3]."

        if frames.dtype != np.uint8:
            frames = frames.astype(np.uint8, copy=False)

        return self._predict_fn(frames)

    def predict_frames(self, frames: np.ndarray, batch_size: int = 32, verbose: bool = True):
        assert len(frames.shape) == 4 and frames.shape[1:] == self._input_size, \
            "[TransNetV2] Input shape must be [frames, height, width, 3]."

        n_frames = len(frames)

        no_padded_frames_start = 25
        no_padded_frames_end = 25 + 50 - (n_frames % 50 if n_frames % 50 != 0 else 50)

        start_frame = np.expand_dims(frames[0], 0)
        end_frame = np.expand_dims(frames[-1], 0)

        padded_inputs = np.concatenate(
            [start_frame] * no_padded_frames_start
            + [frames]
            + [end_frame] * no_padded_frames_end,
            axis=0
        )

        single_predictions = []
        all_predictions = []

        batch_windows = []
        processed = 0

        ptr = 0
        while ptr + 100 <= len(padded_inputs):
            window = padded_inputs[ptr:ptr + 100]
            batch_windows.append(window)
            ptr += 50

            if len(batch_windows) == batch_size:
                self._process_window_batch(
                    batch_windows,
                    single_predictions,
                    all_predictions
                )

                processed += len(batch_windows) * 50
                batch_windows = []

                if verbose:
                    print(
                        "\r[TransNetV2] Processing video frames {}/{}".format(
                            min(processed, n_frames), n_frames
                        ),
                        end=""
                    )

        if len(batch_windows) > 0:
            self._process_window_batch(
                batch_windows,
                single_predictions,
                all_predictions
            )

            processed += len(batch_windows) * 50

            if verbose:
                print(
                    "\r[TransNetV2] Processing video frames {}/{}".format(
                        min(processed, n_frames), n_frames
                    ),
                    end=""
                )

        if verbose:
            print("")

        single_frame_pred = np.concatenate(single_predictions, axis=0)
        all_frames_pred = np.concatenate(all_predictions, axis=0)

        return (
            single_frame_pred[:n_frames],
            all_frames_pred[:n_frames]
        )

    def _process_window_batch(self, batch_windows, single_predictions, all_predictions):
        batch = np.stack(batch_windows, axis=0)

        single_frame_pred, all_frames_pred = self.predict_raw(batch)

        single_frame_pred = single_frame_pred.numpy()[:, 25:75, 0]
        all_frames_pred = all_frames_pred.numpy()[:, 25:75, 0]

        single_predictions.append(single_frame_pred.reshape(-1))
        all_predictions.append(all_frames_pred.reshape(-1))

    def predict_video(
        self,
        video_fn: str,
        batch_size: int = 32,
        frame_stride: int = 1,
        frame_offset: int = 0
    ):
        try:
            import ffmpeg
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Install ffmpeg and ffmpeg-python before using predict_video."
            )

        if frame_stride <= 0:
            raise ValueError("frame_stride must be a positive integer.")

        if frame_offset < 0 or frame_offset >= frame_stride:
            raise ValueError("frame_offset must satisfy 0 <= frame_offset < frame_stride.")

        print("[TransNetV2] Extracting frames from {}".format(video_fn))

        stream = ffmpeg.input(video_fn)

        if frame_stride > 1:
            # Keep frames whose original index n satisfies:
            # (n - frame_offset) % frame_stride == 0
            #
            # For frame_stride=2, frame_offset=0:
            # keep n = 0, 2, 4, 6, ...
            select_expr = f"not(mod(n-{frame_offset},{frame_stride}))"
            stream = stream.filter("select", select_expr)

        stream = stream.filter("scale", 48, 27, flags="bilinear")

        try:
            video_stream, err = (
                stream
                .output(
                    "pipe:",
                    format="rawvideo",
                    pix_fmt="rgb24",
                    vcodec="rawvideo",
                    vsync="0"
                )
                .global_args("-v", "error", "-nostdin")
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as exc:
            error_message = exc.stderr.decode("utf-8", errors="ignore")
            raise RuntimeError(
                f"FFmpeg failed while processing {video_fn}:\n{error_message}"
            ) from exc

        video = np.frombuffer(video_stream, np.uint8).reshape([-1, 27, 48, 3])

        return (
            video,
            *self.predict_frames(video, batch_size=batch_size)
        )
    
    @staticmethod
    def predictions_to_scenes(predictions: np.ndarray, threshold: float = 0.5):
        predictions = (predictions > threshold).astype(np.uint8)

        scenes = []
        t, t_prev, start = -1, 0, 0
        for i, t in enumerate(predictions):
            if t_prev == 1 and t == 0:
                start = i
            if t_prev == 0 and t == 1 and i != 0:
                scenes.append([start, i])
            t_prev = t
        if t == 0:
            scenes.append([start, i])

        # just fix if all predictions are 1
        if len(scenes) == 0:
            return np.array([[0, len(predictions) - 1]], dtype=np.int32)

        return np.array(scenes, dtype=np.int32)

    @staticmethod
    def visualize_predictions(frames: np.ndarray, predictions):
        from PIL import Image, ImageDraw

        if isinstance(predictions, np.ndarray):
            predictions = [predictions]

        ih, iw, ic = frames.shape[1:]
        width = 25

        # pad frames so that length of the video is divisible by width
        # pad frames also by len(predictions) pixels in width in order to show predictions
        pad_with = width - len(frames) % width if len(frames) % width != 0 else 0
        frames = np.pad(frames, [(0, pad_with), (0, 1), (0, len(predictions)), (0, 0)])

        predictions = [np.pad(x, (0, pad_with)) for x in predictions]
        height = len(frames) // width

        img = frames.reshape([height, width, ih + 1, iw + len(predictions), ic])
        img = np.concatenate(np.split(
            np.concatenate(np.split(img, height), axis=2)[0], width
        ), axis=2)[0, :-1]

        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)

        # iterate over all frames
        for i, pred in enumerate(zip(*predictions)):
            x, y = i % width, i // width
            x, y = x * (iw + len(predictions)) + iw, y * (ih + 1) + ih - 1

            # we can visualize multiple predictions per single frame
            for j, p in enumerate(pred):
                color = [0, 0, 0]
                color[(j + 1) % 3] = 255

                value = round(p * (ih - 1))
                if value != 0:
                    draw.line((x + j, y, x + j, y - value), fill=tuple(color), width=1)
        return img


def main():
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=str, nargs="+", help="path to video files to process")
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="path to TransNet V2 weights"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="batch size for TransNetV2 inference"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="save a png file with prediction visualization for each extracted video"
    )

    args = parser.parse_args()

    model = TransNetV2(args.weights)

    for file in args.files:
        if os.path.exists(file + ".predictions.txt") or os.path.exists(file + ".scenes.txt"):
            print(
                f"[TransNetV2] {file}.predictions.txt or {file}.scenes.txt already exists. "
                f"Skipping video {file}.",
                file=sys.stderr
            )
            continue

        video_frames, single_preds, all_preds = model.predict_video(
            file,
            batch_size=args.batch_size
        )

        predictions = np.stack([single_preds, all_preds], axis=1)
        np.savetxt(file + ".predictions.txt", predictions, fmt="%.6f")

        scenes = model.predictions_to_scenes(single_preds)
        np.savetxt(file + ".scenes.txt", scenes, fmt="%d")

        if args.visualize:
            if os.path.exists(file + ".vis.png"):
                print(
                    f"[TransNetV2] {file}.vis.png already exists. "
                    f"Skipping visualization of video {file}.",
                    file=sys.stderr
                )
                continue

            pil_image = model.visualize_predictions(
                video_frames,
                predictions=(single_preds, all_preds)
            )
            pil_image.save(file + ".vis.png")

if __name__ == "__main__":
    main()
