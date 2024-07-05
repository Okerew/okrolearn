from okrolearn.src.okrolearn import Tensor, np
import cv2
from PIL import Image as PILImage


class Image(Tensor):
    def __init__(self, data, requires_grad=True):
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
        gray = cv2.cvtColor(self.data, cv2.COLOR_BGR2GRAY)
        return Image(gray, self.requires_grad)

    def resize(self, size):
        resized = cv2.resize(self.data, size)
        return Image(resized, self.requires_grad)

    def rotate(self, angle):
        rows, cols = self.data.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated = cv2.warpAffine(self.data, M, (cols, rows))
        return Image(rotated, self.requires_grad)

    def flip(self, flip_code):
        flipped = cv2.flip(self.data, flip_code)
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
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        if len(data.shape) == 3:
            data = np.expand_dims(data, axis=-1)
        elif len(data.shape) != 4:
            raise ValueError("Video data must be 3D (frames, height, width) or 4D (frames, height, width, channels)")

        # Ensure channel dimension is last
        if data.shape[-1] not in [1, 3, 4]:
            data = np.moveaxis(data, 1, -1)

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
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in self.data]
        return Video(np.array(gray_frames), self.requires_grad)

    def resize(self, size):
        resized_frames = [cv2.resize(frame, size) for frame in self.data]
        return Video(np.array(resized_frames), self.requires_grad)

    def rotate(self, angle):
        rows, cols = self.data[0].shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated_frames = [cv2.warpAffine(frame, M, (cols, rows)) for frame in self.data]
        return Video(np.array(rotated_frames), self.requires_grad)

    def save(self, filename, fps=30):
        """Save the Video tensor to a file."""
        Decoder.save_video(self, filename, fps)

    def display(self):
        """Display the Video tensor."""
        Decoder.display_video(self)


class ComputerVision:
    @staticmethod
    def detect_edges(image, low_threshold, high_threshold):
        if not isinstance(image, Image):
            raise ValueError("Input must be an Image object")

        edges = cv2.Canny(image.data, low_threshold, high_threshold)
        return Image(edges)

    @staticmethod
    def detect_corners(image, max_corners=100, quality_level=0.01, min_distance=10):
        if not isinstance(image, Image):
            raise ValueError("Input must be an Image object")

        gray = image.to_grayscale().data
        corners = cv2.goodFeaturesToTrack(gray, max_corners, quality_level, min_distance)
        return corners

    @staticmethod
    def find_contours(image):
        if not isinstance(image, Image):
            raise ValueError("Input must be an Image object")

        gray = image.to_grayscale().data
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    @staticmethod
    def optical_flow(prev_image, next_image):
        if not isinstance(prev_image, Image) or not isinstance(next_image, Image):
            raise ValueError("Inputs must be Image objects")

        prev_gray = prev_image.to_grayscale().data
        next_gray = next_image.to_grayscale().data
        flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        return flow

    @staticmethod
    def optical_flow_video(video):
        if not isinstance(video, Video):
            raise ValueError("Input must be a Video object")

        prev_frame = video.data[0]
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        flows = []

        for frame in video.data[1:]:
            next_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flows.append(flow)
            prev_gray = next_gray

        return np.array(flows)


class Decoder:
    @staticmethod
    def tensor_to_image(tensor):
        """
        Convert a tensor to a PIL Image.

        Args:
        tensor (Tensor): Input tensor with shape (channels, height, width) or (height, width)

        Returns:
        PIL.Image: Decoded image
        """
        if not isinstance(tensor, Tensor):
            raise ValueError("Input must be a Tensor")

        # Ensure the tensor is on CPU and detached from the computation graph
        data = tensor.data.cpu().detach().numpy() if hasattr(tensor.data, 'cpu') else tensor.data

        # Normalize the data to the 0-255 range
        data = np.clip(data, 0, 1)
        data = (data * 255).astype(np.uint8)

        # Reshape if necessary
        if len(data.shape) == 3:
            # (channels, height, width) -> (height, width, channels)
            data = np.transpose(data, (1, 2, 0))
        elif len(data.shape) != 2:
            raise ValueError("Tensor must have 2 or 3 dimensions")

        # Convert to PIL Image
        if len(data.shape) == 2 or data.shape[2] == 1:
            return PILImage.fromarray(data.squeeze(), mode='L')
        elif data.shape[2] == 3:
            return PILImage.fromarray(data, mode='RGB')
        elif data.shape[2] == 4:
            return PILImage.fromarray(data, mode='RGBA')
        else:
            raise ValueError("Unsupported number of channels")

    @staticmethod
    def save_image(tensor, filename):
        """
        Save a tensor as an image file.

        Args:
        tensor (Tensor): Input tensor
        filename (str): Output filename
        """
        img = Decoder.tensor_to_image(tensor)
        img.save(filename)

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
        if not isinstance(tensor, Video):
            raise ValueError("Input must be a Video object")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (tensor.data.shape[2], tensor.data.shape[1]))

        for frame in tensor.data:
            out.write(frame)

        out.release()

    @staticmethod
    def display_video(tensor):
        """
        Display a video tensor.

        Args:
        tensor (Video): Input video tensor
        """
        if not isinstance(tensor, Video):
            raise ValueError("Input must be a Video object")

        for frame in tensor.data:
            cv2.imshow('Video', frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    @staticmethod
    def batch_to_grid(tensor, nrow=8, padding=2):
        """
        Make a grid of images from a batch of tensors.

        Args:
        tensor (Tensor): Input tensor of shape (batch_size, channels, height, width)
        nrow (int): Number of images displayed in each row of the grid
        padding (int): Amount of padding between images

        Returns:
        Tensor: Grid of images
        """
        if not isinstance(tensor, Tensor):
            raise ValueError("Input must be a Tensor")

        if len(tensor.data.shape) != 4:
            raise ValueError("Input tensor must be 4D (batch_size, channels, height, width)")

        batch_size, channels, height, width = tensor.data.shape

        # Compute grid size
        ncol = (batch_size + nrow - 1) // nrow

        # Create empty grid
        grid = np.zeros((channels, height * nrow + padding * (nrow - 1),
                         width * ncol + padding * (ncol - 1)))

        # Fill the grid
        for idx, img in enumerate(tensor.data):
            i = idx % nrow
            j = idx // nrow
            grid[:, i * (height + padding):i * (height + padding) + height,
            j * (width + padding):j * (width + padding) + width] = img

        return Tensor(grid)
