import io
import base64
from IPython.core.display import display, HTML

def fix_layout(width:int=95):
    display(HTML("<style>.container { width:" + str(width) + "% !important; }</style>"))


def display_video(path):
    video = io.open(path, 'r+b').read()
    encoded = base64.b64encode(video)
    return HTML(data='''<video alt="test" controls>
                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                 </video>'''.format(encoded.decode('ascii')))
