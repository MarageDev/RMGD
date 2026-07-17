# code from Mahé DUVAL
import sys
from pathlib import Path

# Add parent directory to path for imports and file pathsto work from the Demos folder more easily
sys.path.insert(0, str(Path(__file__).parent.parent))

# UI Imports
import gradio as gr
from demo.resources.theme import *

from demo.demo_niftypp import demo_niftypp

with gr.Blocks() as demo:
    gr.HTML(HTML_LOGO_HEADER)
    
    niftypp_header = gr.HTML(HTML_HEADER_NIFTYPP, visible=True)
    
    gr.HTML(HTML_AUTHORS)
    with gr.Tabs(selected="a"):
        demo_niftypp()
    gr.HTML(HTML_FOOTER)

demo.queue().launch(css_paths="./demo/resources/style.css",head=HTML_CUSTOM_HEAD, theme=theme)