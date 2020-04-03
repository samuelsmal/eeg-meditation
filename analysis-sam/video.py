import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mne import viz
import numpy as np
import cv2

# Use Agg backend for canvas
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def index_to_time(x, time_index, step_size=1):
    """Helper function to add the axis labels"""
    if (x < 0 or x * step_size >= len(time_index)):
        return ''
    
    seconds = time_index[int(x*step_size)].total_seconds()
    return f"{int(seconds/60)}\' {seconds/60:.2f}\""

def create_video(data, times_index_to_plot, pos, file_name):
    # create OpenCV video writer
    video = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc('A','V','C','1'), 1, (250, 250))
    bandpower_over_time_index = data.index
    
    # loop over your images
    for idx, time_index in enumerate(times_index_to_plot):
        fig = plt.figure()
        viz.plot_topomap(data.iloc[time_index, :].T.values, 
                         sphere=1.,
                         pos=pos,
                         sensors=False,
                         show_names=True,
                         show=False,
                         names=data.columns)

        plt.title(index_to_time(times_index_to_plot[idx], bandpower_over_time_index))
        fig.canvas.draw()

        mat = np.array(fig.canvas.renderer._renderer)
        mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)

        # write frame to video
        video.write(mat)
        plt.close()

    # close video writer
    cv2.destroyAllWindows()
    video.release()