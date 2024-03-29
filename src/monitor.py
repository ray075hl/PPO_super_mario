import subprocess as sp
import numpy as np
import platform


system_info = platform.platform()
print(system_info)
if system_info[:3] == 'Win':
    FFMPEG_BIN = 'ffmpeg.exe'
else:
    FFMPEG_BIN = 'ffmpeg'


class Monitor:
    def __init__(self, width, height, save_path='./saved_video/output_video.mp4'):
        
        self.command = [ FFMPEG_BIN,
                '-y', # (optional) overwrite output file if it exists
                '-f', 'rawvideo',
                '-vcodec','rawvideo',
                '-s', '{}X{}'.format(width, height), # size of one frame
                '-pix_fmt', 'rgb24',
                '-r', '24', # frames per second
                '-i', '-', # The imput comes from a pipe
                '-an', # Tells FFMPEG not to expect any audio
                '-vcodec', 'mpeg4',
                save_path ]
        try:
            self.pipe = sp.Popen( self.command, stdin=sp.PIPE, stderr=sp.PIPE)
        except FileNotFoundError:
            print('*'*50)
            print('Can not find ffmpeg, please set right path of it.')
            print('*'*50)		
    def record(self, image_array):
        self.pipe.stdin.write( image_array.tostring() )


# unit test
if __name__ == '__main__':
    width, height = 640, 480
    monitor = Monitor(width, height)
    for _ in range(100):
        frame = np.zeros((height, width, 3))
        monitor.record(frame)
