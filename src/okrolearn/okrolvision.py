from src.okrolearn.okrolearn import Tensor, np
import cv2
from PIL import Image as PILImage
import numpy as numpy


class Image(Tensor):
    def __init__(self, data, requires_grad=True):
        if isinstance(data, np.ndarray):
            data = data.get()
        super().__init__(data, requires_grad)
        if len(self.data.shape) not in [2, 3]:
            raise ValueError("Image data must be 2D (grayscale) or 3D (color)")

    @classmethod
    def from_file(cls, file_path):
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError(f"Could not read image from {file_path}")
        return cls(img)

    def to_grayscale(self):
        if len(self.data.shape) == 2:
            return self
        data = self.data.get() if isinstance(self.data, np.ndarray) else self.data
        gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        return Image(gray, self.requires_grad)

    def resize(self, size):
        data = self.data.get() if isinstance(self.data, np.ndarray) else self.data
        resized = cv2.resize(data, size)
        return Image(resized, self.requires_grad)

    def rotate(self, angle):
        data = self.data.get() if isinstance(self.data, np.ndarray) else self.data
        rows, cols = data.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated = cv2.warpAffine(data, M, (cols, rows))
        return Image(rotated, self.requires_grad)

    def flip(self, flip_code):
        data = self.data.get() if isinstance(self.data, np.ndarray) else self.data
        flipped = cv2.flip(data, flip_code)
        return Image(flipped, self.requires_grad)

    def to_pil_image(self):
        """Convert the Image tensor to a PIL Image."""
        return Decoder.tensor_to_image(self)

    def save(self, filename):
        """Save the Image tensor to a file."""
        Decoder.save_image(self, filename)

    def display(self):
        """Display the Image tensor."""
        Decoder.display_image(self)


class Video(Tensor):
    def __init__(self, data, requires_grad=True):
        if isinstance(data, np.ndarray):
            data = data.get()
        if not isinstance(data, numpy.ndarray):
            data = numpy.array(data)

        if len(data.shape) == 3:
            data = numpy.expand_dims(data, axis=-1)
        elif len(data.shape) != 4:
            raise ValueError("Video data must be 3D (frames, height, width) or 4D (frames, height, width, channels)")

        # Ensure channel dimension is last
        if data.shape[-1] not in [1, 3, 4]:
            data = numpy.moveaxis(data, 1, -1)

        super().__init__(data, requires_grad)

    @classmethod
    def from_file(cls, file_path):
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video from {file_path}")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        return cls(np.array(frames), requires_grad=True)

    def to_grayscale(self):
        data = self.data.get() if isinstance(self.data, np.ndarray) else self.data
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in data]
        return Video(numpy.array(gray_frames), self.requires_grad)

    def resize(self, size):
        data = self.data.get() if isinstance(self.data, np.ndarray) else self.data
        resized_frames = [cv2.resize(frame, size) for frame in data]
        return Video(numpy.array(resized_frames), self.requires_grad)

    def rotate(self, angle):
        data = self.data.get() if isinstance(self.data, np.ndarray) else self.data
        rows, cols = data[0].shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated_frames = [cv2.warpAffine(frame, M, (cols, rows)) for frame in data]
        return Video(numpy.array(rotated_frames), self.requires_grad)

    def save(self, filename, fps=30):
        """Save the Video tensor to a file."""
        Decoder.save_video(self, filename, fps)

    def display(self, fps=30):
        """Display the Video tensor."""
        Decoder.display_video(self, fps)


class ComputerVision:
    @staticmethod
    def detect_edges(image, low_threshold, high_threshold):
        if not isinstance(image, Image):
            raise ValueError("Input must be an Image object")
        data = image.data.get() if isinstance(image.data, np.ndarray) else image.data
        edges = cv2.Canny(data, low_threshold, high_threshold)
        return Image(edges)

    @staticmethod
    def detect_corners(image, max_corners=100, quality_level=0.01, min_distance=10):
        if not isinstance(image, Image):
            raise ValueError("Input must be an Image object")
        gray = image.to_grayscale().data
        gray = gray.get() if isinstance(gray, np.ndarray) else gray
        corners = cv2.goodFeaturesToTrack(gray, max_corners, quality_level, min_distance)
        return corners

    @staticmethod
    def find_contours(image):
        if not isinstance(image, Image):
            raise ValueError("Input must be an Image object")
        gray = image.to_grayscale().data
        gray = gray.get() if isinstance(gray, np.ndarray) else gray
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    @staticmethod
    def optical_flow(prev_image, next_image):
        if not isinstance(prev_image, Image) or not isinstance(next_image, Image):
            raise ValueError("Inputs must be Image objects")
        prev_gray = prev_image.to_grayscale().data
        next_gray = next_image.to_grayscale().data
        prev_gray = prev_gray.get() if isinstance(prev_gray, np.ndarray) else prev_gray
        next_gray = next_gray.get() if isinstance(next_gray, np.ndarray) else next_gray
        flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        return flow

    @staticmethod
    def optical_flow_video(video):
        if not isinstance(video, Video):
            raise ValueError("Input must be a Video object")
        data = video.data.get() if isinstance(video.data, np.ndarray) else video.data
        prev_frame = data[0]
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        flows = []

        for frame in data[1:]:
            next_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flows.append(flow)
            prev_gray = next_gray

        return numpy.array(flows)


class Decoder:
    @staticmethod
    def tensor_to_image(tensor):
        """
        Convert a tensor to a PIL Image, automatically reducing to 3 channels if necessary.

        Args:
        tensor (Tensor): Input tensor with shape (channels, height, width), (height, width),
                         or (batch_size, channels, height, width)

        Returns:
        PIL.Image: Decoded image
        """
        if not isinstance(tensor, Tensor):
            raise ValueError("Input must be a Tensor")

        # Ensure the tensor is on CPU and detached from the computation graph
        if isinstance(tensor.data, np.ndarray):
            data = tensor.data.get()
        else:
            data = tensor.data

        # Handle different input shapes
        if len(data.shape) == 4:  # (batch_size, channels, height, width)
            if data.shape[0] != 1:
                raise ValueError(f"Expected batch size of 1, but got {data.shape[0]}")
            data = data.squeeze(0)
        elif len(data.shape) not in [2, 3]:
            raise ValueError(f"Tensor must have 2, 3, or 4 dimensions, but got {len(data.shape)}")

        # If the input is 2D, expand it to 3D
        if len(data.shape) == 2:
            data = data[numpy.newaxis, :, :]

        # Reduce to 3 channels if necessary
        if data.shape[0] > 3:
            data = Decoder._reduce_channels(tensor)
        elif data.shape[0] < 3:
            data = Decoder._expand_channels(data)

        # Normalize the data to the 0-255 range
        data = numpy.clip(data, 0, 1)
        data = (data * 255).astype(numpy.uint8)

        # (channels, height, width) -> (height, width, channels)
        data = numpy.transpose(data, (1, 2, 0))

        return PILImage.fromarray(data, mode='RGB')

    @staticmethod
    def _reduce_channels(tensor):
        """Reduce the number of channels to 3 using PCA."""
        c, h, w = tensor.data.shape
        reshaped = Tensor(tensor.data.reshape(c, -1).T)
        reduced = reshaped.pca(n_components=3)
        return reduced.data.T.reshape(3, h, w)

    @staticmethod
    def _expand_channels(data):
        """Expand the number of channels to 3 by duplicating."""
        c, h, w = data.shape
        if c == 1:
            return numpy.repeat(data, 3, axis=0)
        elif c == 2:
            return numpy.concatenate([data, data[0:1]], axis=0)

    @staticmethod
    def save_image(tensor, filename):
        """
        Save a tensor as an image file.

        Args:
        tensor (Tensor): Input tensor
        filename (str): Output filename
        """
        try:
            img = Decoder.tensor_to_image(tensor)
            img.save(filename)
        except Exception as e:
            print(f"Error saving image: {e}")
            print(f"Tensor shape: {tensor.data.shape}")
            raise

    @staticmethod
    def display_image(tensor):
        """
        Display a tensor as an image.

        Args:
        tensor (Tensor): Input tensor
        """
        img = Decoder.tensor_to_image(tensor)
        img.show()

    @staticmethod
    def save_video(tensor, filename, fps=30):
        """
        Save a video tensor as a video file.

        Args:
        tensor (Video): Input video tensor
        filename (str): Output filename
        fps (int): Frames per second
        """
        if not isinstance(tensor, Tensor):
            raise ValueError("Input must be a Tensor object")

        try:
            data = tensor.data.get() if isinstance(tensor.data, np.ndarray) else tensor.data

            if len(data.shape) != 4:
                raise ValueError(f"Expected 4D tensor (frames, channels, height, width), but got shape {data.shape}")

            num_frames, num_channels, height, width = data.shape

            # Convert to 3 channels if necessary
            if num_channels != 3:
                data = Decoder._convert_video_channels(data)

            fourcc = cv2.VideoWriter(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

            for frame in data:
                # Convert frame to uint8 and correct shape (height, width, channels)
                frame = (numpy.clip(frame, 0, 1) * 255).astype(numpy.uint8)
                frame = numpy.transpose(frame, (1, 2, 0))

                # OpenCV uses BGR format
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                out.write(frame)

            out.release()
        except Exception as e:
            print(f"Error saving video: {e}")
            print(f"Tensor shape: {tensor.data.shape}")
            raise

    @staticmethod
    def _convert_video_channels(data):
        """Convert video data to 3 channels."""
        num_frames, num_channels, height, width = data.shape
        if num_channels > 3:
            # Use PCA to reduce to 3 channels
            reshaped = data.reshape(num_frames * height * width, num_channels)
            tensor = Tensor(reshaped)
            reduced = tensor.pca(n_components=3)
            return reduced.data.reshape(num_frames, 3, height, width)
        elif num_channels == 1:
            # Repeat grayscale channel 3 times
            return numpy.repeat(data, 3, axis=1)
        elif num_channels == 2:
            # Add a third channel (could be more sophisticated)
            return numpy.concatenate([data, data[:, :1, :, :]], axis=1)
        else:
            return data

    @staticmethod
    def display_video(tensor, fps=30):
        """
        Display a video tensor.

        Args:
        tensor (Video): Input video tensor
        fps (int): Frames per second
        """
        if not isinstance(tensor, Tensor):
            raise ValueError("Input must be a Tensor object")

        try:
            data = tensor.data.get() if isinstance(tensor.data, np.ndarray) else tensor.data

            if len(data.shape) != 4:
                raise ValueError(f"Expected 4D tensor (frames, channels, height, width), but got shape {data.shape}")

            num_frames, num_channels, height, width = data.shape

            # Convert to 3 channels if necessary
            if num_channels != 3:
                data = Decoder._convert_video_channels(data)

            for frame in data:
                # Convert frame to uint8 and correct shape (height, width, channels)
                frame = (numpy.clip(frame, 0, 1) * 255).astype(numpy.uint8)
                frame = numpy.transpose(frame, (1, 2, 0))

                # OpenCV uses BGR format
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                cv2.imshow('Video', frame)
                if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                    break

            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error displaying video: {e}")
            print(f"Tensor shape: {tensor.data.shape}")
            raise

    @staticmethod
    def batch_to_grid(tensor, nrow=8, padding=2):
        if not isinstance(tensor, Tensor):
            raise ValueError("Input must be a Tensor")

        if len(tensor.data.shape) != 4:
            raise ValueError("Input tensor must be 4D (batch_size, channels, height, width)")

        if isinstance(tensor.data, np.ndarray):
            data = tensor.data.get()
        else:
            data = tensor.data

        batch_size, channels, height, width = data.shape

        # Compute grid size
        ncol = (batch_size + nrow - 1) // nrow

        # Create empty grid
        grid = numpy.zeros((channels, height * nrow + padding * (nrow - 1),
                            width * ncol + padding * (ncol - 1)))

        # Fill the grid
        for idx, img in enumerate(data):
            i = idx % nrow
            j = idx // nrow
            grid[:, i * (height + padding):i * (height + padding) + height,
            j * (width + padding):j * (width + padding) + width] = img

        return Tensor(grid)
