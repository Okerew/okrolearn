from src.okrolearn.okrolvision import Video, ComputerVision
video = Video.from_file('video.mp4')
gray_video = video.to_grayscale()
resized_video = video.resize((640, 480))
rotated_video = video.rotate(45)
resized_video.save('resized_video.mp4')
rotated_video.display()

optical_flow = ComputerVision.optical_flow_video(video)
